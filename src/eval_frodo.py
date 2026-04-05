import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftConfig, PeftModel

from frodo import FRODO, FRODOConfig

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

TEST_FILES = {
    "strategyqa": "strategyqa/data/test_gpt3_responses.jsonl",
    "obqa":       "obqa/data/test_gpt3_responses.jsonl",
    "quarel":     "quarel/data/test_gpt3_responses.jsonl",
    "qasc":       "qasc/data/test_gpt3_responses.jsonl",
}

TRAIN_CF_FILES = {
    "strategyqa": "strategyqa/data/train_gpt3_responses_counterfactual.json",
    "obqa":       "obqa/data/train_gpt3_responses_counterfactual.json",
    "quarel":     "quarel/data/train_gpt3_responses_counterfactual.json",
    "qasc":       "qasc/data/train_gpt3_responses_output_v2.json",
}


def load_test_data(dataset):
    with open(DATA_ROOT / TEST_FILES[dataset]) as f:
        raw = json.load(f)
    seen = {}
    for entry in raw.values():
        q = entry["question"]
        if q not in seen:
            seen[q] = entry["gold_label"]
    return [{"question": q, "gold_label": g} for q, g in seen.items()]


def load_few_shot_examples(dataset, n_shots, seed=42):
    with open(DATA_ROOT / TRAIN_CF_FILES[dataset]) as f:
        data = json.load(f)
    rng = random.Random(seed)
    pool = [e for e in data if e.get("predicted_rationale") and e.get("gold_label")]
    rng.shuffle(pool)
    return pool[:n_shots]


def build_few_shot_prompt(question, examples):
    parts = []
    for ex in examples:
        parts.append(
            f"Q: {ex['question']}\n"
            f"Reasoning: {ex['predicted_rationale']}\n"
            f"Answer: {ex['gold_label']}"
        )
    parts.append(f"Q: {question}\nReasoning:")
    return "\n\n".join(parts)


def normalize(s):
    s = s.strip().lower().rstrip(".")
    for prefix in ["the answer is ", "answer: ", "final answer: "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def is_correct(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    if g in ("yes", "no") and p.startswith(g):
        return True
    if g.startswith("(") and g in p:
        return True
    return False


def load_tokenizer(model_dir, fallback_name):
    tok_dir = model_dir / "tokenizer"
    if tok_dir.exists():
        tok = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tok = AutoTokenizer.from_pretrained(fallback_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def select_model_kwargs(device):
    if not device.startswith("cuda"):
        return {}
    return {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}


def load_model_from_dir(model_path, device, fallback_base=None):
    model_path = Path(model_path)
    model_kw = select_model_kwargs(device)

    adapter_cfg = model_path / "adapter_config.json"
    if adapter_cfg.exists():
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_name = peft_cfg.base_model_name_or_path if peft_cfg.base_model_name_or_path else fallback_base
        if base_name is None:
            raise ValueError(f"Unable to determine base model for adapter at {model_path}")
        base_cfg = AutoConfig.from_pretrained(base_name)
        is_enc_dec = getattr(base_cfg, "is_encoder_decoder", False)
        AutoModelCls = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM
        base_model = AutoModelCls.from_pretrained(base_name, **model_kw)
        model = PeftModel.from_pretrained(base_model, model_path)
        return model.to(device), is_enc_dec, base_name

    model_cfg = AutoConfig.from_pretrained(model_path)
    is_enc_dec = getattr(model_cfg, "is_encoder_decoder", False)
    AutoModelCls = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM
    model = AutoModelCls.from_pretrained(model_path, **model_kw).to(device)
    return model, is_enc_dec, str(model_path)


def generate_text(model, tokenizer, prompt, device, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kw = dict(**encoded, do_sample=False)
    if getattr(model.config, "is_encoder_decoder", False):
        gen_kw["max_length"] = encoded["input_ids"].shape[1] + max_new_tokens
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
    out = model.generate(**gen_kw)
    if not getattr(model.config, "is_encoder_decoder", False):
        out = out[:, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def clean_answer(text):
    answer = text.strip().split("\n")[0]
    for prefix in ["answer:", "final answer:"]:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="strategyqa", choices=list(TEST_FILES.keys()))
    p.add_argument("--model_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_shots", type=int, default=3)
    p.add_argument("--inference_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_reasoning_tokens", type=int, default=128)
    p.add_argument("--max_answer_tokens", type=int, default=16)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    inference_model, is_enc_dec, base_name = load_model_from_dir(model_dir / "inference_model", args.device)
    tokenizer = load_tokenizer(model_dir, base_name)

    has_reasoning_model = (model_dir / "reasoning_model").exists()
    inference_only = args.inference_only or (not has_reasoning_model)
    frodo = None
    if not inference_only:
        reasoning_model, _, _ = load_model_from_dir(model_dir / "reasoning_model", args.device, fallback_base=base_name)
        gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        config = FRODOConfig(
            max_length=args.max_length,
            use_ddp=False,
            local_rank=gpu_idx,
            is_encoder_decoder=is_enc_dec,
        )
        frodo = FRODO(inference_model, reasoning_model, config).to(config.device)
        frodo.eval()
    else:
        inference_model.eval()
        print("Running in inference-only mode")

    few_shot_examples = []
    if args.n_shots > 0:
        few_shot_examples = load_few_shot_examples(args.dataset, args.n_shots)
        print(f"Using {len(few_shot_examples)}-shot prompt")

    test_data = load_test_data(args.dataset)
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    correct, total = 0, 0
    for item in tqdm(test_data, desc="Evaluating"):
        if few_shot_examples:
            prompt = build_few_shot_prompt(item["question"], few_shot_examples)
        else:
            prompt = item["question"]
        if inference_only:
            reasoning = generate_text(
                inference_model,
                tokenizer,
                prompt,
                args.device,
                max_new_tokens=args.max_reasoning_tokens,
            )
            answer_prompt = f"{prompt}\nReasoning: {reasoning}\nAnswer:"
            answer = clean_answer(
                generate_text(
                    inference_model,
                    tokenizer,
                    answer_prompt,
                    args.device,
                    max_new_tokens=args.max_answer_tokens,
                )
            )
        else:
            reasoning, answer = frodo.generate_reasoning_and_answer(prompt, tokenizer)
        hit = is_correct(answer, item["gold_label"])
        correct += int(hit)
        total += 1
        if total <= 5 or (not hit and total <= 20):
            print(f"  Q: {item['question']}")
            print(f"  Gold: {item['gold_label']}  Pred: {answer}  {'OK' if hit else 'WRONG'}")
            print()

    acc = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()
