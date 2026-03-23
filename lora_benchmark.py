"""
LoRA Fine-Tuning Throughput Benchmark
Benchmarks LoRA training speed (tokens/sec) for a ~8B model on the current GPU.
Outputs a JSON results file and a throughput PNG plot.
"""

import argparse
import datetime
import json
import math
import platform
import statistics
import subprocess
import threading
import time

import matplotlib
matplotlib.use("Agg")  # headless - no display required
import matplotlib.pyplot as plt
import numpy as np

import torch

if not torch.cuda.is_available():
    build = getattr(torch.version, "cuda", None)
    msg = (
        "No CUDA GPU detected. "
        + (f"PyTorch was built with CUDA {build} but no device is visible - check your drivers."
           if build else
           "PyTorch appears to be a CPU-only build - reinstall with a CUDA-enabled wheel.")
    )
    raise SystemExit(f"[ERROR] {msg}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning throughput benchmark")
    p.add_argument("--machine-label", default="unknown",
                   help="Human-readable label for this machine (used in output filenames)")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (required for gated models)")
    p.add_argument("--devices", default=None,
                   help="Comma-separated GPU indices to use (e.g. '0,1,2'). Default: all available GPUs.")
    p.add_argument("--lora-rank",    type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len",  type=int,   default=512,
                   help="Reduce to 256 if OOM on low-VRAM GPUs")
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=2)
    p.add_argument("--warmup-steps", type=int,   default=5)
    p.add_argument("--bench-steps",  type=int,   default=50)
    p.add_argument("--dtype",        default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dataset-id",   default="yahma/alpaca-cleaned")
    p.add_argument("--max-samples",  type=int,   default=2000)
    return p.parse_args()


# ── Device resolution ─────────────────────────────────────────────────────────

def resolve_devices(args) -> list:
    n = torch.cuda.device_count()
    if n == 0:
        return []
    if args.devices is None:
        return list(range(n))
    devices = [int(d.strip()) for d in args.devices.split(",")]
    for d in devices:
        if d >= n:
            raise SystemExit(f"[ERROR] Device {d} not available (found {n} GPU(s))")
    return devices


# ── GPU info ──────────────────────────────────────────────────────────────────

def gpu_info(machine_label: str, devices: list) -> dict:
    gpu_list = []
    for i in devices:
        props = torch.cuda.get_device_properties(i)
        gpu_list.append({"index": i, "name": props.name,
                         "vram_gb": round(props.total_memory / 1024 ** 3, 1)})
    total_vram_gb = round(sum(g["vram_gb"] for g in gpu_list), 1)
    gpu_label = ", ".join(f"[{g['index']}] {g['name']}" for g in gpu_list)

    id_str = ",".join(str(d) for d in devices)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", f"--id={id_str}",
             "--query-gpu=driver_version,power.limit",
             "--format=csv,noheader,nounits"], text=True
        ).strip().split("\n")[0].split(", ")
        driver, power = smi[0], smi[1]
    except Exception:
        driver, power = "N/A", "N/A"

    return {
        "machine":      machine_label,
        "gpu":          gpu_label,
        "gpu_count":    len(devices),
        "gpu_list":     gpu_list,
        "cuda_version": torch.version.cuda,
        "driver":       driver,
        "vram_gb":      total_vram_gb,
        "power_w":      power,
        "torch":        torch.__version__,
        "python":       platform.python_version(),
    }


# ── GPU utilisation sampler ───────────────────────────────────────────────────

class _GPUSampler:
    """Polls nvidia-smi every `interval` seconds during the benchmark loop.

    Captures:
      utilization.gpu    - SM (compute) utilisation %
      utilization.memory - memory-controller utilisation % (proxy for BW %)
    """
    def __init__(self, devices: list, interval: float = 0.5):
        self.interval = interval
        self._id_str  = ",".join(str(d) for d in devices)
        self._sm:  list = []
        self._mem: list = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                rows = subprocess.check_output(
                    ["nvidia-smi", f"--id={self._id_str}",
                     "--query-gpu=utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    text=True,
                ).strip().split("\n")
                sm_vals  = [float(r.split(", ")[0]) for r in rows if r.strip()]
                mem_vals = [float(r.split(", ")[1]) for r in rows if r.strip()]
                if sm_vals:
                    self._sm.append(sum(sm_vals) / len(sm_vals))
                    self._mem.append(sum(mem_vals) / len(mem_vals))
            except Exception:
                pass

    def summary(self) -> dict:
        def stats(lst):
            return {"mean": round(statistics.mean(lst), 1),
                    "peak": round(max(lst), 1)} if lst else {"mean": None, "peak": None}
        return {"sm_util": stats(self._sm), "mem_ctrl_util": stats(self._mem)}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args, info: dict, devices: list):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    vram_gb = info["vram_gb"]
    print(f"VRAM: {vram_gb} GB  ->  LoRA ({args.dtype.upper()})")

    print(f"Loading {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Unified-memory GPUs (e.g. DGX Spark GB10, NVLink-C2C) share physical
    # memory with the CPU. NVML reports 0 bytes of dedicated VRAM on these
    # devices, which breaks Accelerate's infer_auto_device_map. Use an
    # explicit single-device map instead.
    # Discrete GPUs (A100, H100, …) use device_map="auto" with a max_memory
    # budget so layers are spread only across the selected devices.
    is_unified = torch.cuda.get_device_properties(devices[0]).is_integrated
    if is_unified:
        _device_map = {"": devices[0]}
        _max_memory = None
    else:
        n_total = torch.cuda.device_count()
        _max_memory = {i: "0GiB" for i in range(n_total)}
        for i in devices:
            _max_memory[i] = torch.cuda.get_device_properties(i).total_memory
        _device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        dtype=torch_dtype,
        device_map=_device_map,
        max_memory=_max_memory,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_data(args, tokenizer):
    from datasets import load_dataset

    raw = load_dataset(args.dataset_id, split="train")
    raw = raw.select(range(min(args.max_samples, len(raw))))
    print(f"Loaded {len(raw)} samples from {args.dataset_id}")

    def format_sample(ex):
        instruction = ex.get("instruction", "").strip()
        inp         = ex.get("input", "").strip()
        output      = ex.get("output", "").strip()
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return {"text": text}

    dataset   = raw.map(format_sample, remove_columns=raw.column_names)
    tokenized = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True,
                             max_length=args.max_seq_len, padding="max_length"),
        batched=True, remove_columns=["text"],
    )
    tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"].copy()})
    tokenized.set_format(type="torch")
    print(f"Dataset ready: {len(tokenized)} samples, max_seq_len={args.max_seq_len}")
    return tokenized


# ── Training loop ─────────────────────────────────────────────────────────────

def run_benchmark(args, model, dataset, devices: list):
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # pin_memory speeds up PCIe transfers to discrete GPUs; on unified-memory
    # systems (e.g. DGX Spark GB10) the CPU and GPU share physical memory so
    # pinning pages adds overhead with no benefit.
    _pin_memory = not torch.cuda.get_device_properties(devices[0]).is_integrated
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=_pin_memory)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    total_steps = args.warmup_steps + args.bench_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Bytes occupied by all model parameters (used to estimate memory traffic)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    model.train()
    device = next(model.parameters()).device

    step         = 0
    bench_start  = None
    bench_tokens = 0
    step_times   = []

    tokens_per_step = args.batch_size * args.max_seq_len
    print(f"\nWarmup: {args.warmup_steps} steps | Benchmark: {args.bench_steps} steps")
    print(f"Batch: {args.batch_size} | Seq len: {args.max_seq_len} | Tokens/step: {tokens_per_step:,}")
    print("-" * 60)

    sampler = _GPUSampler(devices=devices, interval=0.5)

    for batch in loader:
        if step >= total_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        t0      = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step == args.warmup_steps:
            bench_start = time.perf_counter()
            sampler.start()
            print("  Warmup done. Starting benchmark ...")

        if step >= args.warmup_steps:
            bench_tokens += input_ids.numel()
            step_times.append(t1 - t0)

        if step % 10 == 0:
            tps       = input_ids.numel() / (t1 - t0)
            vram_used = sum(torch.cuda.memory_allocated(i) for i in devices) / 1024 ** 3 if torch.cuda.is_available() else 0
            phase     = "[BENCH]" if step >= args.warmup_steps else "[WARM ]"
            print(f"  {phase} step {step:>3} | loss={outputs.loss.item():.4f} "
                  f"| {tps:>8,.0f} tok/s | VRAM {vram_used:.1f} GB")

        step += 1

    sampler.stop()
    if bench_start is None:
        raise SystemExit("[ERROR] Benchmark never started - warmup_steps exceeds the number of available batches.")
    bench_wall = time.perf_counter() - bench_start
    print("-" * 60)
    print("Benchmark loop complete.")
    return step_times, bench_tokens, bench_wall, sampler.summary(), param_bytes


# ── Results ───────────────────────────────────────────────────────────────────

def summarise(args, info: dict,
              step_times, bench_tokens, bench_wall, gpu_samples: dict, param_bytes: int,
              devices: list):
    tokens_per_step = args.batch_size * args.max_seq_len
    per_step_tps    = [tokens_per_step / t for t in step_times]
    mean_tps        = statistics.mean(per_step_tps)
    median_tps      = statistics.median(per_step_tps)
    peak_tps        = max(per_step_tps)
    total_tps       = bench_tokens / bench_wall

    vram_peak_gb = sum(torch.cuda.max_memory_allocated(i) for i in devices) / 1024 ** 3 if torch.cuda.is_available() else 0

    corpus_tokens   = 500_000_000
    epoch_secs      = corpus_tokens / total_tps
    epoch_hours     = epoch_secs / 3600
    epoch_days      = epoch_secs / 86400
    five_epoch_days = epoch_days * 5

    # Memory bandwidth estimate.
    # Lower bound: each training step reads all model weights at least twice
    # (once in the forward pass, once in the backward pass).  Gradient writes
    # and activation traffic are excluded, so the true figure will be higher.
    param_gb        = param_bytes / 1024 ** 3
    mean_step_s     = statistics.mean(step_times)
    est_bw_gb_s     = (param_gb * 2) / mean_step_s  # GB/s lower bound

    sm  = gpu_samples.get("sm_util",      {"mean": None, "peak": None})
    mem = gpu_samples.get("mem_ctrl_util", {"mean": None, "peak": None})

    def fmt_pct(v):
        return f"{v:.1f}%" if v is not None else "N/A"

    print("\n" + "=" * 60)
    print(f"  RESULTS - {args.machine_label}")
    print("=" * 60)
    print(f"  Model:               {args.model_id}")
    print(f"  GPU:                 {info['gpu']}")
    print(f"  VRAM total:          {info['vram_gb']} GB")
    print(f"  VRAM peak (used):    {vram_peak_gb:.1f} GB")
    print(f"  Precision:           {args.dtype.upper()}")
    print(f"  Seq len / batch:     {args.max_seq_len} / {args.batch_size}")
    print(f"  LoRA rank:           {args.lora_rank}")
    print()
    print(f"  Mean tok/s:          {mean_tps:>10,.0f}")
    print(f"  Median tok/s:        {median_tps:>10,.0f}")
    print(f"  Peak tok/s:          {peak_tps:>10,.0f}")
    print(f"  Wall-clock tok/s:    {total_tps:>10,.0f}")
    print()
    print(f"  --- Memory throughput ---")
    print(f"  Model params:        {param_gb:.2f} GB  ({param_bytes / 1e9:.2f} B params × dtype)")
    print(f"  Est. BW (fwd+bwd):   {est_bw_gb_s:.1f} GB/s  (lower bound - weight reads only)")
    print(f"  SM utilisation:      mean {fmt_pct(sm['mean'])}  peak {fmt_pct(sm['peak'])}")
    print(f"  Mem-ctrl utilisation:mean {fmt_pct(mem['mean'])}  peak {fmt_pct(mem['peak'])}")
    print()
    print(f"  --- Projection (500M token corpus) ---")
    print(f"  1 epoch:             {epoch_hours:.1f} hours  ({epoch_days:.2f} days)")
    print(f"  5 epochs:            {five_epoch_days:.1f} days")
    print("=" * 60)

    return dict(
        per_step_tps=per_step_tps,
        mean_tps=mean_tps, median_tps=median_tps,
        peak_tps=peak_tps, total_tps=total_tps,
        vram_peak_gb=vram_peak_gb,
        epoch_hours=epoch_hours, five_epoch_days=five_epoch_days,
        param_gb=param_gb,
        est_mem_bw_gb_s=round(est_bw_gb_s, 1),
        mean_sm_util_pct=sm["mean"],
        peak_sm_util_pct=sm["peak"],
        mean_mem_ctrl_util_pct=mem["mean"],
        peak_mem_ctrl_util_pct=mem["peak"],
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_plot(args, info: dict, metrics: dict):
    per_step_tps = metrics["per_step_tps"]
    mean_tps     = metrics["mean_tps"]
    steps_x      = list(range(len(per_step_tps)))
    rolling      = np.convolve(per_step_tps, np.ones(5) / 5, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_x, per_step_tps, alpha=0.35, color="steelblue", label="Per-step")
    ax.plot(range(len(rolling)), rolling, color="steelblue", linewidth=2, label="5-step avg")
    ax.axhline(mean_tps, color="tomato", linestyle="--",
               label=f"Mean: {mean_tps:,.0f} tok/s")
    ax.set_xlabel("Benchmark Step")
    ax.set_ylabel("Tokens / second")
    ax.set_title(f"LoRA Training Throughput - {args.machine_label} ({info['gpu']})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    safe_label = args.machine_label.replace(" ", "_")
    fname = f"throughput_{safe_label}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot saved: {fname}")
    return fname


# ── JSON export ───────────────────────────────────────────────────────────────

def save_json(args, info: dict, metrics: dict):
    safe_label = args.machine_label.replace(" ", "_")
    fname = f"results_{safe_label}.json"
    results = {
        "timestamp":       datetime.datetime.now().isoformat(),
        "machine":         args.machine_label,
        "model":           args.model_id,
        "gpu":             info["gpu"],
        "vram_total_gb":   info["vram_gb"],
        "vram_peak_gb":    round(metrics["vram_peak_gb"], 2),
        "precision":       args.dtype,
        "lora_rank":       args.lora_rank,
        "seq_len":         args.max_seq_len,
        "batch_size":      args.batch_size,
        "bench_steps":     args.bench_steps,
        "mean_tps":               round(metrics["mean_tps"], 1),
        "median_tps":             round(metrics["median_tps"], 1),
        "peak_tps":               round(metrics["peak_tps"], 1),
        "wall_tps":               round(metrics["total_tps"], 1),
        "epoch_hours":            round(metrics["epoch_hours"], 2),
        "five_epoch_days":        round(metrics["five_epoch_days"], 2),
        "param_gb":               round(metrics["param_gb"], 2),
        "est_mem_bw_gb_s":        metrics["est_mem_bw_gb_s"],
        "mean_sm_util_pct":       metrics["mean_sm_util_pct"],
        "peak_sm_util_pct":       metrics["peak_sm_util_pct"],
        "mean_mem_ctrl_util_pct": metrics["mean_mem_ctrl_util_pct"],
        "peak_mem_ctrl_util_pct": metrics["peak_mem_ctrl_util_pct"],
    }
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {fname}")
    return fname


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(" LoRA Throughput Benchmark")
    print("=" * 60)

    devices = resolve_devices(args)
    print(f"Using GPU(s): {devices}")

    info = gpu_info(args.machine_label, devices)
    print(json.dumps(info, indent=2))

    model, tokenizer = load_model(args, info, devices)
    dataset = load_data(args, tokenizer)
    step_times, bench_tokens, bench_wall, gpu_samples, param_bytes = run_benchmark(args, model, dataset, devices)
    metrics = summarise(args, info, step_times, bench_tokens, bench_wall, gpu_samples, param_bytes, devices)
    save_plot(args, info, metrics)
    save_json(args, info, metrics)


if __name__ == "__main__":
    main()
