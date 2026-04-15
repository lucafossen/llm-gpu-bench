#!/usr/bin/env bash
# LoRA Benchmark – portable bootstrap + runner
# Usage: bash run_benchmark.sh [--machine-label "A100 80GB"] [--model-id MODEL] [--hf-token TOKEN] [--devices 0,1,2]
#
# Drop this directory on any Linux/macOS GPU server and run this script.
# It will install Miniconda locally (./miniconda3), create a conda env, install
# all Python deps, run the benchmark, and print the path to the result files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$SCRIPT_DIR/miniconda3"
ENV_NAME="lora_bench"
PYTHON_VERSION="3.11"

# ── Parse arguments ─────────────────────────────────────────────────────────
MACHINE_LABEL="$(hostname -s)"
BACKEND="hf"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine-label) MACHINE_LABEL="$2"; shift 2 ;;
        --backend)       BACKEND="$2"; EXTRA_ARGS+=("--backend" "$2"); shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Install Miniconda (local, non-root) ──────────────────────────────────────
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    echo "==> Miniconda not found — installing to $CONDA_DIR"
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Linux-x86_64)   INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
        Linux-aarch64)  INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
        Darwin-x86_64)  INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh" ;;
        Darwin-arm64)   INSTALLER="Miniconda3-latest-MacOSX-arm64.sh" ;;
        *) echo "Unsupported OS/arch: $OS-$ARCH"; exit 1 ;;
    esac
    INSTALLER_PATH="/tmp/$INSTALLER"
    if command -v wget &>/dev/null; then
        wget -q --show-progress "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$INSTALLER_PATH"
    else
        curl -fL "https://repo.anaconda.com/miniconda/$INSTALLER" -o "$INSTALLER_PATH"
    fi
    bash "$INSTALLER_PATH" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER_PATH"
    echo "==> Miniconda installed."
fi

# ── Activate conda ───────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ── Accept conda ToS (required since Anaconda repo policy update) ────────────
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ── Create env ───────────────────────────────────────────────────────────────
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "==> Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

# ── Detect CUDA version and select PyTorch wheel index ───────────────────────
# Returns either a stable index URL or "nightly:<url>" for pre-release wheels.
detect_torch_index_url() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "https://download.pytorch.org/whl/cpu"; return
    fi
    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    if [[ -z "$cuda_ver" ]]; then
        echo "https://download.pytorch.org/whl/cpu"; return
    fi
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    # Check GPU compute capability — Blackwell (sm_12x) requires nightly + cu128
    local sm_major=0
    sm_major=$(python - 2>/dev/null <<'PYEOF'
import subprocess, re, sys
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True).strip().split("\n")[0]
    print(out.split(".")[0])
except Exception:
    print("0")
PYEOF
)

    if [[ "$sm_major" -ge 12 ]]; then
        # DGX Spark GB10 (sm_121, CUDA 13) needs cu130; desktop Blackwell (CUDA 12.8) uses nightly cu128
        if [[ "$major" -ge 13 ]]; then
            echo "https://download.pytorch.org/whl/cu130"; return
        fi
        echo "nightly:https://download.pytorch.org/whl/nightly/cu128"; return
    fi

    if   [[ "$major" -ge 13 ]];                        then echo "https://download.pytorch.org/whl/cu130"
    elif [[ "$major" -eq 12 && "$minor" -ge 4 ]];      then echo "https://download.pytorch.org/whl/cu124"
    elif [[ "$major" -eq 12 && "$minor" -ge 1 ]];      then echo "https://download.pytorch.org/whl/cu121"
    elif [[ "$major" -eq 11 && "$minor" -ge 8 ]];      then echo "https://download.pytorch.org/whl/cu118"
    else echo "https://download.pytorch.org/whl/cpu"
    fi
}

# ── Install Python packages ──────────────────────────────────────────────────
echo "==> Installing Python packages ..."
pip install -q --upgrade pip

TORCH_INDEX_RAW="$(detect_torch_index_url)"
echo "==> Detected PyTorch index: $TORCH_INDEX_RAW"
if [[ "$TORCH_INDEX_RAW" == nightly:* ]]; then
    TORCH_INDEX="${TORCH_INDEX_RAW#nightly:}"
    # torchvision excluded: nightly removed VideoReader, which breaks datasets' torch formatter
    pip install -q --pre torch --index-url "$TORCH_INDEX"
else
    TORCH_INDEX="$TORCH_INDEX_RAW"
    pip install -q torch --index-url "$TORCH_INDEX"
fi

# Verify CUDA is accessible after install
if ! python - <<'EOF'
import torch, sys
if torch.cuda.is_available():
    print(f"==> PyTorch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
else:
    print("ERROR: torch.cuda.is_available() is False after install.", file=sys.stderr)
    print(f"       torch version: {torch.__version__}, CUDA build: {torch.version.cuda}", file=sys.stderr)
    sys.exit(1)
EOF
then
    echo ""
    echo "HINT: The installed PyTorch may not match your CUDA version."
    echo "      Driver CUDA: $(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)"
    echo "      Try running the script again — the index URL selection may need adjustment."
    exit 1
fi

pip install -q transformers peft datasets accelerate trl \
               matplotlib pandas numpy

if [[ "$BACKEND" == "nemo" ]]; then
    echo "==> Installing nemo-automodel (required for --backend nemo) ..."
    pip install -q nemo-automodel
fi

# ── Run benchmark ────────────────────────────────────────────────────────────
echo ""
echo "==> Running benchmark (machine: '${MACHINE_LABEL}') ..."
echo ""

cd "$SCRIPT_DIR"
python lora_benchmark.py --machine-label "$MACHINE_LABEL" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# ── Print result locations ───────────────────────────────────────────────────
SAFE_LABEL="${MACHINE_LABEL// /_}"
echo ""
echo "==> Done!  Results written to:"
echo "      $SCRIPT_DIR/results_${SAFE_LABEL}.json"
echo "      $SCRIPT_DIR/throughput_${SAFE_LABEL}.png"
echo "      Backend: $BACKEND"
