import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="strategyqa", choices=list(TEST_FILES.keys()))
    p.add_argument("--model_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_shots", type=int, default=3)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cfg = AutoConfig.from_pretrained(model_dir / "inference_model")
    is_enc_dec = getattr(model_cfg, "is_encoder_decoder", False)
    AutoModelCls = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM

    inference_model = AutoModelCls.from_pretrained(model_dir / "inference_model").to(args.device)
    reasoning_model = AutoModelCls.from_pretrained(model_dir / "reasoning_model").to(args.device)

    gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    config = FRODOConfig(max_length=args.max_length, use_ddp=True, local_rank=gpu_idx,
                         is_encoder_decoder=is_enc_dec)
    frodo = FRODO(inference_model, reasoning_model, config)
    frodo = frodo.to(config.device)
    frodo.eval()

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
