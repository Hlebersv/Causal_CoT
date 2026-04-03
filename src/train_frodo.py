import argparse, copy, gc, json, os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from frodo import (
    FRODOConfig,
    InferenceModule,
    ReasoningModule,
    DPODataset,
    ReasoningDataset,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
)

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


def make_scheduler(optimizer, dataset_size, config, grad_accum_steps):
    batches_per_epoch = max((dataset_size + config.batch_size - 1) // config.batch_size, 1)
    if config.use_ddp:
        batches_per_epoch = max((dataset_size + (config.batch_size * config.world_size) - 1) // (config.batch_size * config.world_size), 1)
    optimizer_steps = max((batches_per_epoch * config.num_epochs + grad_accum_steps - 1) // grad_accum_steps, 1)
    return get_linear_schedule_with_warmup(optimizer, config.warmup_steps, optimizer_steps)


def maybe_enable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        # Required for checkpointing on most decoder-only HF models.
        model.config.use_cache = False


def maybe_apply_lora(model, use_lora, is_enc_dec, r, alpha, dropout, target_modules):
    if not use_lora:
        return model
    task_type = TaskType.SEQ_2_SEQ_LM if is_enc_dec else TaskType.CAUSAL_LM
    targets = [m.strip() for m in target_modules.split(",") if m.strip()] if target_modules else None
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=task_type,
    )
    return get_peft_model(model, lora_cfg)


def cleanup_phase():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_phase(module, module_name, dataset, config, optimizer, scheduler, writer, num_epochs, grad_accum_steps=1, global_step=0):
    is_inference = (module_name == "inference")
    module.train()
    loader = make_loader(dataset, config.batch_size, config.use_ddp, config.world_size)

    for epoch in range(num_epochs):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(loader):
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
                    answer_ids=batch["answer_ids"].to(dev), answer_mask=batch["answer_mask"].to(dev),
                    counterfactual_reasoning_ids=cf_ids.to(dev) if cf_ids is not None else None,
                    counterfactual_reasoning_mask=cf_mask.to(dev) if cf_mask is not None else None,
                )
            loss = out["loss"] / grad_accum_steps
            loss.backward()
            should_step = ((step_idx + 1) % grad_accum_steps == 0) or (step_idx + 1 == len(loader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                global_step += 1
            total_loss += out["loss"].item()
            n += 1
            if is_main_process() and writer:
                writer.add_scalar(f"{module_name}/micro_step_loss", out["loss"].item(), epoch * len(loader) + step_idx)
                if should_step:
                    writer.add_scalar(f"{module_name}/step_loss", out["loss"].item(), global_step)
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
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--find_unused_parameters", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )
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
        print(
            f"dataset={args.dataset}  model={args.model}  ddp={use_ddp} x{world_size}  "
            f"bs={args.batch_size}  accum={args.grad_accum_steps}  lr={args.lr}  lora={args.use_lora}"
        )

    dpo_data, reasoning_data = load_dataset(args.dataset)
    if args.max_samples:
        dpo_data = dpo_data[:args.max_samples]
        reasoning_data = reasoning_data[:args.max_samples]
    if is_main_process():
        print(f"dpo={len(dpo_data)}  reasoning={len(reasoning_data)}")

    tok_kw = {"token": args.hf_token} if args.hf_token else {}
    model_cfg = AutoConfig.from_pretrained(args.model, **tok_kw)
    is_enc_dec = getattr(model_cfg, "is_encoder_decoder", False)
    config.is_encoder_decoder = is_enc_dec

    AutoModelCls = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kw = dict(tok_kw)
    if not is_enc_dec and torch.cuda.is_available():
        model_kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if is_main_process():
        print(f"arch={'enc-dec' if is_enc_dec else 'causal'}  pad_token={tokenizer.pad_token}")

    writer = None
    if is_main_process() and args.tensorboard:
        log_dir = args.tensorboard_log_dir or f"runs/{args.dataset}"
        writer = SummaryWriter(log_dir=log_dir)

    out = Path(args.output_dir) / args.dataset
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)

    dpo_ds = DPODataset(dpo_data, tokenizer, max_length=config.max_length)
    if is_main_process():
        print("=== Phase 1: Inference (DPO) ===")
    inference_model = AutoModelCls.from_pretrained(args.model, **model_kw)
    inference_model = maybe_apply_lora(
        inference_model,
        args.use_lora,
        is_enc_dec,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_target_modules,
    )
    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(inference_model)
    inference_module = InferenceModule(inference_model, config).to(config.device)
    ref_model = None
    if not args.use_lora:
        ref_model = copy.deepcopy(inference_module.model)
        inference_module.set_reference_model(ref_model)
    elif is_main_process():
        print("using base model (disabled adapters) as DPO reference")
    if use_ddp:
        inference_module.model = DDP(
            inference_module.model,
            device_ids=[config.local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
    if args.use_lora and is_main_process() and hasattr(inference_module._unwrap(), "print_trainable_parameters"):
        inference_module._unwrap().print_trainable_parameters()
    inf_opt = AdamW((p for p in inference_module.parameters() if p.requires_grad), lr=config.learning_rate)
    inf_sched = make_scheduler(inf_opt, len(dpo_ds), config, args.grad_accum_steps)
    gs = train_phase(
        inference_module,
        "inference",
        dpo_ds,
        config,
        inf_opt,
        inf_sched,
        writer,
        config.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
    )
    if use_ddp and dist.is_initialized():
        dist.barrier()
    if is_main_process():
        inference_module._unwrap().save_pretrained(out / "inference_model")
    if ref_model is not None:
        del ref_model
    del inference_module, inference_model, inf_opt, inf_sched
    cleanup_phase()
    if use_ddp and dist.is_initialized():
        dist.barrier()

    rea_ds = ReasoningDataset(reasoning_data, tokenizer, max_length=config.max_length)
    if is_main_process():
        print("=== Phase 2: Reasoning ===")
    reasoning_model = AutoModelCls.from_pretrained(args.model, **model_kw)
    reasoning_model = maybe_apply_lora(
        reasoning_model,
        args.use_lora,
        is_enc_dec,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_target_modules,
    )
    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(reasoning_model)
    reasoning_module = ReasoningModule(reasoning_model, config).to(config.device)
    if use_ddp:
        reasoning_module.model = DDP(
            reasoning_module.model,
            device_ids=[config.local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
    if args.use_lora and is_main_process() and hasattr(reasoning_module._unwrap(), "print_trainable_parameters"):
        reasoning_module._unwrap().print_trainable_parameters()
    rea_opt = AdamW((p for p in reasoning_module.parameters() if p.requires_grad), lr=config.learning_rate)
    rea_sched = make_scheduler(rea_opt, len(rea_ds), config, args.grad_accum_steps)
    train_phase(
        reasoning_module,
        "reasoning",
        rea_ds,
        config,
        rea_opt,
        rea_sched,
        writer,
        config.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        global_step=gs,
    )
    if use_ddp and dist.is_initialized():
        dist.barrier()
    if is_main_process():
        reasoning_module._unwrap().save_pretrained(out / "reasoning_model")
        tokenizer.save_pretrained(out / "tokenizer")
        if writer:
            writer.close()
        print(f"saved to {out}")
    del reasoning_module, reasoning_model, rea_opt, rea_sched
    cleanup_phase()

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
