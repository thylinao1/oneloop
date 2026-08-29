#!/bin/bash
#SBATCH --job-name=amex-safety-injection
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=xgpj0
#SBATCH --output=%x-%j.out

set -euo pipefail

# All caches to node-local scratch; the home quota is shared.
SCRATCH=/tmp/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export VLLM_CACHE_ROOT="$SCRATCH/vllm"
export PYTHONUNBUFFERED=1

echo "[job] node $(hostname), gpus ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h /tmp | tail -1

source ~/miniconda3/bin/activate csa-vllm

# Fail-fast capability guard, in the shape job_protection.sbatch uses. Qwen3-32B
# in bfloat16 is 61.02 GiB of weights. nvidia-smi is NOT a safe check here: on
# the a100-40 nodes it names an 80 GB card while handing the process a 39.25 GiB
# partition, which is what killed jobs 752533, 752536 and 752537. Ask torch what
# the process actually got.
python - <<'PY' || exit 3
import sys
import torch
n = torch.cuda.device_count()
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"[job] torch sees {n} device(s), device 0 has {gib:.2f} GiB", flush=True)
sys.exit(0 if gib >= 70.0 else 3)
PY

WORK=$HOME/amex-oneloop/safety
cd "$WORK"

python -u injection_run.py \
  --corpus injection_corpus.json \
  --out safety_injection_raw.json \
  --model Qwen/Qwen3-32B \
  --tp 1 \
  --gpu-mem-util 0.94

echo "[job] done:"
ls -la safety_injection_raw.json
