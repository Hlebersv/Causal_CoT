import argparse, copy, json, os
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from frodo import FRODO, FRODOConfig, DPODataset, ReasoningDataset, setup_ddp, cleanup_ddp, is_main_process

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "strategyqa": ("strategyqa/data/train_gpt3_responses.jsonl", "strategyqa/data/train_gpt3_responses_counterfactual.json"),
    "obqa":       ("obqa/data/train_gpt3_responses.jsonl",       "obqa/data/train_gpt3_responses_counterfactual.json"),
    "quarel":     ("quarel/data/train_gpt3_responses.jsonl",     "quarel/data/train_gpt3_responses_counterfactual.json"),
    "qasc":       ("qasc/data/train_gpt3_responses.jsonl",       "qasc/data/train_gpt3_responses_output_v2.json"),
}


def load_dataset(name):
    train_path, cf_path = DATASETS[name]
    with open(DATA_ROOT / train_path) as f:
        raw = json.load(f)
    with open(DATA_ROOT / cf_path) as f:
        cf_list = json.load(f)

    cf_map = {e["question"]: e["counterfactual"] for e in cf_list}

    by_q = {}
    for entry in raw.values():
        q = entry["question"]
        if q not in by_q:
            by_q[q] = {"question": q, "gold_label": entry["gold_label"], "rationales": []}
        if entry.get("predicted_rationale"):
            by_q[q]["rationales"].append((entry["predicted_rationale"], entry["predicted_label"]))

    dpo_data, reasoning_data = [], []
    for item in by_q.values():
        if not item["rationales"]:
            continue
        best = next((r for r, l in item["rationales"] if l.strip().lower() == item["gold_label"].strip().lower()), item["rationales"][0][0])
        q = item["question"]
        if q in cf_map:
            dpo_data.append({"question": q, "preferred_reasoning": best, "dispreferred_reasoning": cf_map[q]})
            reasoning_data.append({"question": q, "reasoning": best, "answer": item["gold_label"], "counterfactual_reasoning": cf_map[q]})
        else:
            reasoning_data.append({"question": q, "reasoning": best, "answer": item["gold_label"]})

    return dpo_data, reasoning_data


def make_loader(dataset, batch_size, use_ddp, world_size, shuffle=True):
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=int(os.environ.get("RANK", 0)), shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_phase(frodo, module_name, dataset, config, optimizer, scheduler, writer, num_epochs, global_step=0):
    is_inference = (module_name == "inference")
    module = frodo.inference_module if is_inference else frodo.reasoning_module
    module.train()
    loader = make_loader(dataset, config.batch_size, config.use_ddp, config.world_size)

    for epoch in range(num_epochs):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss, n = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            dev = config.device
            if is_inference:
                out = module(
                    input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                    preferred_ids=batch["preferred_ids"].to(dev), preferred_mask=batch["preferred_mask"].to(dev),
                    dispreferred_ids=batch["dispreferred_ids"].to(dev), dispreferred_mask=batch["dispreferred_mask"].to(dev),
                )
            else:
                cf_ids = batch.get("counterfactual_reasoning_ids")
                cf_mask = batch.get("counterfactual_reasoning_mask")
                out = module(
                    question_ids=batch["question_ids"].to(dev), question_mask=batch["question_mask"].to(dev),
                    reasoning_ids=batch["reasoning_ids"].to(dev), reasoning_mask=batch["reasoning_mask"].to(dev),
                    answer_ids=batch["answer_ids"].to(dev),
                    counterfactual_reasoning_ids=cf_ids.to(dev) if cf_ids is not None else None,
                    counterfactual_reasoning_mask=cf_mask.to(dev) if cf_mask is not None else None,
                )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
            n += 1
            global_step += 1
            if is_main_process() and writer:
                writer.add_scalar(f"{module_name}/step_loss", loss.item(), global_step)
                if not is_inference:
                    writer.add_scalar(f"{module_name}/lm_loss", out["lm_loss"], global_step)
                    writer.add_scalar(f"{module_name}/ie_loss", out["ie_loss"], global_step)
                    writer.add_scalar(f"{module_name}/margin_loss", out["margin_loss"], global_step)

        if is_main_process():
            avg = total_loss / max(n, 1)
            print(f"  [{module_name}] epoch {epoch+1}/{num_epochs}  loss={avg:.4f}")
            if writer:
                writer.add_scalar(f"{module_name}/epoch_loss", avg, epoch)

    return global_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="strategyqa", choices=list(DATASETS.keys()))
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lambda_lm", type=float, default=1.0)
    p.add_argument("--lambda_ie", type=float, default=1.0)
    p.add_argument("--lambda_margin", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--tensorboard", action="store_true", default=True)
    p.add_argument("--tensorboard_log_dir", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank, world_size = 0, 1
    if use_ddp:
        local_rank = setup_ddp()
        world_size = int(os.environ["WORLD_SIZE"])

    config = FRODOConfig(
        model_name=args.model, max_length=args.max_length, learning_rate=args.lr,
        batch_size=args.batch_size, num_epochs=args.num_epochs, warmup_steps=args.warmup_steps,
        beta=args.beta, lambda_lm=args.lambda_lm, lambda_ie=args.lambda_ie,
        lambda_margin=args.lambda_margin, margin=args.margin,
        use_ddp=use_ddp, local_rank=local_rank, world_size=world_size,
    )

    if is_main_process():
        print(f"dataset={args.dataset}  model={args.model}  ddp={use_ddp} x{world_size}  bs={args.batch_size}  lr={args.lr}")

    dpo_data, reasoning_data = load_dataset(args.dataset)
    if args.max_samples:
        dpo_data = dpo_data[:args.max_samples]
        reasoning_data = reasoning_data[:args.max_samples]
    if is_main_process():
        print(f"dpo={len(dpo_data)}  reasoning={len(reasoning_data)}")

    tok_kw = {"token": args.hf_token} if args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    inference_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **tok_kw)
    reasoning_model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **tok_kw)

    frodo = FRODO(inference_model, reasoning_model, config).to(config.device)
    ref_model = copy.deepcopy(frodo.inference_module.model)
    frodo.inference_module.set_reference_model(ref_model)
    if use_ddp:
        frodo.wrap_ddp()

    writer = None
    if is_main_process() and args.tensorboard:
        log_dir = args.tensorboard_log_dir or f"runs/{args.dataset}"
        writer = SummaryWriter(log_dir=log_dir)

    dpo_ds = DPODataset(dpo_data, tokenizer, max_length=config.max_length)
    inf_opt = AdamW(frodo.inference_module.parameters(), lr=config.learning_rate)
    inf_steps = (len(dpo_ds) // (config.batch_size * world_size) + 1) * config.num_epochs
    inf_sched = get_linear_schedule_with_warmup(inf_opt, config.warmup_steps, inf_steps)

    if is_main_process():
        print("=== Phase 1: Inference (DPO) ===")
    gs = train_phase(frodo, "inference", dpo_ds, config, inf_opt, inf_sched, writer, config.num_epochs)

    rea_ds = ReasoningDataset(reasoning_data, tokenizer, max_length=config.max_length)
    rea_opt = AdamW(frodo.reasoning_module.parameters(), lr=config.learning_rate)
    rea_steps = (len(rea_ds) // (config.batch_size * world_size) + 1) * config.num_epochs
    rea_sched = get_linear_schedule_with_warmup(rea_opt, config.warmup_steps, rea_steps)

    if is_main_process():
        print("=== Phase 2: Reasoning ===")
    train_phase(frodo, "reasoning", rea_ds, config, rea_opt, rea_sched, writer, config.num_epochs, gs)

    if is_main_process():
        out = Path(args.output_dir) / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        inf_m = frodo.inference_module._unwrap()
        rea_m = frodo.reasoning_module._unwrap()
        inf_m.save_pretrained(out / "inference_model")
        rea_m.save_pretrained(out / "reasoning_model")
        tokenizer.save_pretrained(out / "tokenizer")
        if writer:
            writer.close()
        print(f"saved to {out}")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
