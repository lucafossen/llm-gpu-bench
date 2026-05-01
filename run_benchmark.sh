#!/usr/bin/env bash
# LoRA Benchmark – portable bootstrap + runner
# Usage: bash run_benchmark.sh [--machine-label "A100 80GB"] [--model-id MODEL] [--hf-token TOKEN] [--devices 0,1,2]
#
# Drop this directory on any Linux/macOS GPU server and run this script.
# It will install uv (if needed), create a local .venv with Python 3.12,
# install all Python deps, run the benchmark, and print the result file paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="3.12"

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

# ── Ensure uv is available ───────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:$PATH"
    else
        echo "==> Installing uv ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:$PATH"
    fi
fi

# ── Detect CUDA version and select PyTorch wheel index ───────────────────────
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

    # Check GPU compute capability — Blackwell (sm_12x) requires cu130 or nightly
    local sm_major=0
    sm_major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | cut -d. -f1 || echo "0")

    if [[ "$sm_major" -ge 12 ]]; then
        # DGX Spark GB10 (sm_121, CUDA 13) uses stable cu130
        # Desktop Blackwell (CUDA 12.8) uses nightly cu128
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

# ── Create venv (once) ───────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating uv environment (Python ${PYTHON_VERSION}) ..."
    uv python install "$PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    echo "==> Environment created at $VENV_DIR"
fi

# ── Activate ─────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Install nemo-automodel if needed ────────────────────────────────────────
if [[ "$BACKEND" == "nemo" ]] && ! python -c "import nemo_automodel" 2>/dev/null; then
    echo "==> Installing nemo-automodel ..."
    uv pip install nemo-automodel
fi

# ── Ensure CUDA-enabled PyTorch is installed ─────────────────────────────────
TORCH_CUDA_OK=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo "0")
if [[ "$TORCH_CUDA_OK" != "1" ]]; then
    TORCH_INDEX_RAW="$(detect_torch_index_url)"
    echo "==> Installing CUDA-enabled PyTorch (index: ${TORCH_INDEX_RAW#nightly:}) ..."
    if [[ "$TORCH_INDEX_RAW" == nightly:* ]]; then
        uv pip install --reinstall --pre torch --index-url "${TORCH_INDEX_RAW#nightly:}"
    else
        uv pip install --reinstall "torch==2.10.0" --index-url "$TORCH_INDEX_RAW"
    fi
fi

# ── Symlink libcuda.so.1 into Triton's lib dir (idempotent, no sudo needed) ──
# Triton compiles a small C helper at startup and links against libcuda.so.1.
# It searches its own lib dir + /lib/<arch>-linux-gnu, but on many distros the
# file lives elsewhere. We symlink it into the venv (which we own) so Triton
# can always find it without any system-level changes.
find_libcuda() {
    # 1. ldconfig is the authoritative source on Linux
    if command -v ldconfig &>/dev/null; then
        local loc
        loc=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
        [[ -n "$loc" && -e "$loc" ]] && { echo "$loc"; return; }
    fi
    # 2. LD_LIBRARY_PATH
    IFS=: read -ra _ld_dirs <<< "${LD_LIBRARY_PATH:-}"
    for d in "${_ld_dirs[@]}"; do
        [[ -e "$d/libcuda.so.1" ]] && { echo "$d/libcuda.so.1"; return; }
    done
    # 3. Broad search across common locations (no sudo needed — just reading)
    find /usr/lib /lib /usr/local/cuda /usr/local/lib /opt \
         -name "libcuda.so.1" 2>/dev/null | head -1 || true
}

TRITON_LIB="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/triton/backends/nvidia/lib"
if [[ -d "$TRITON_LIB" && ! -e "$TRITON_LIB/libcuda.so.1" ]]; then
    LIBCUDA="$(find_libcuda)"
    if [[ -n "$LIBCUDA" ]]; then
        ln -sf "$LIBCUDA" "$TRITON_LIB/libcuda.so.1"
        echo "==> Linked $LIBCUDA -> $TRITON_LIB/libcuda.so.1"
    else
        echo "WARNING: libcuda.so.1 not found — Triton kernel compilation may fail." >&2
    fi
fi

# ── Install remaining Python deps ────────────────────────────────────────────
uv pip install -q transformers peft datasets accelerate trl matplotlib pandas numpy

# ── Verify CUDA is accessible ────────────────────────────────────────────────
if ! python - <<'EOF'
import torch, sys
if torch.cuda.is_available():
    print(f"==> PyTorch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}")
    sys.exit(0)
else:
    print("ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
    print(f"       torch: {torch.__version__}, CUDA build: {torch.version.cuda}", file=sys.stderr)
    sys.exit(1)
EOF
then
    echo ""
    echo "HINT: Delete .venv/ and rerun — the environment will be rebuilt with CUDA PyTorch."
    echo "      Driver CUDA: $(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)"
    exit 1
fi

# ── Run benchmark ────────────────────────────────────────────────────────────
echo ""
echo "==> Running benchmark (machine: '${MACHINE_LABEL}', backend: '${BACKEND}') ..."
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
