#!/bin/bash

# Set ROOT_DIR via environment variable, or it defaults to the repo root.
# ROOT_DIR is the parent directory used to store models, datasets, and outputs.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export HF_TOKEN="<your_huggingface_token>"
export WANDB_API_KEY="<your_wandb_api_key>"

for i in {1..5}; do
    python src/main_runner.py  --config-name=main_llama_cyber --exp-num=$i
done

