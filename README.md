# lora-benchmark

A portable benchmark for measuring LoRA fine-tuning throughput across different GPU hardware. Run it on a machine, get a JSON report and a plot back.

## What it measures

Training speed (tokens/second) for a ~7B parameter model using LoRA fine-tuning on the [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) instruction dataset. It automatically uses QLoRA (4-bit) on GPUs with less than 30 GB VRAM, and full BF16 LoRA on larger cards.

The benchmark runs 5 warmup steps followed by 50 timed steps, then reports mean, median, peak, and wall-clock throughput, plus projected training time for a 500M token corpus.

## Usage

```bash
git clone https://github.com/lucafossen/lora-benchmark && cd lora-benchmark && bash run_benchmark.sh
```

The script installs Miniconda and all dependencies locally (no root required), then runs the benchmark. When finished, it writes:

- `results_<machine>.json` - full metrics
- `throughput_<machine>.png` - per-step throughput plot

With a custom label:

```bash
bash run_benchmark.sh --machine-label "A100 80GB"
```

To run on specific GPUs (by index):

```bash
bash run_benchmark.sh --devices 0,1
```

By default all available GPUs are used.

## Options

| Flag | Default | Description |
|---|---|---|
| `--machine-label` | hostname | Label used in output filenames and reports |
| `--devices` | all GPUs | Comma-separated GPU indices to use, e.g. `0,1` |
| `--model-id` | `Qwen/Qwen2.5-7B-Instruct` | Any HuggingFace causal LM |
| `--hf-token` | none | Required for gated models (e.g. Llama) |
| `--max-seq-len` | 512 | Reduce to 256 if you get OOM errors |
| `--batch-size` | 4 | Per-device batch size |

## Output fields

Each run produces a `results_<machine>.json` with the following fields:

| Field | Description |
|---|---|
| `mean_tps` | Average tokens/second across all benchmark steps |
| `median_tps` | Median tokens/second, less sensitive to outlier steps than the mean |
| `peak_tps` | Best single-step throughput recorded |
| `wall_tps` | Total tokens divided by total elapsed time - the most realistic number for estimating real training duration |
| `vram_total_gb` | Total VRAM on the GPU |
| `vram_peak_gb` | Peak VRAM actually used during the run - if this is close to total, you're near the memory limit |
| `precision` | `bf16` = full LoRA, `qlora_4bit` = 4-bit quantised (used automatically on GPUs with <30 GB VRAM) |
| `epoch_hours` | Projected hours to train one pass over a 500M token corpus at `wall_tps` |
| `five_epoch_days` | Same projection for five epochs |
| `param_gb` | Total model parameter footprint in GB at the training dtype (e.g. 14 GB for a 7B model in BF16). Useful for sanity-checking `vram_peak_gb` and for estimating memory traffic |
| `est_mem_bw_gb_s` | Estimated memory bandwidth in GB/s, calculated as `param_gb × 2 / mean_step_seconds`. Lower bound only - counts one forward and one backward read of all weights, but excludes activation traffic and gradient writes |
| `mean_sm_util_pct` | Mean GPU compute (SM) utilisation % sampled every 0.5 s during the benchmark. Low values (e.g. <60%) mean the GPU is stalling - on memory latency, the data loader, or CPU-side overhead |
| `peak_sm_util_pct` | Highest SM utilisation sample recorded |
| `mean_mem_ctrl_util_pct` | Mean memory-controller utilisation % (`nvidia-smi utilization.memory`). Measures the fraction of time the memory bus is active - the best no-root proxy for bandwidth saturation. Near 100% = memory-bandwidth-bound; low alongside low SM = bottleneck is elsewhere |
| `peak_mem_ctrl_util_pct` | Highest memory-controller utilisation sample recorded |

## Comparing machines

Collect the `results_*.json` files from each machine.

## Requirements

- Linux or macOS
- NVIDIA GPU with CUDA
- `wget` or `curl` (for the Miniconda installer)
- No root/sudo access needed
