#!/bin/bash

# 从检查点恢复训练脚本
# 使用方法: ./resume_training.sh <checkpoint_path> [additional_iters]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [additional_iters]"
    echo "Example: $0 ./experiments/checkpoints/checkpoint_10000.pt 50000"
    exit 1
fi

CHECKPOINT_PATH="$1"
ADDITIONAL_ITERS=${2:-50000}

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Extract directory information from checkpoint path
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
EXPERIMENT_DIR=$(dirname "$CHECKPOINT_DIR")
DATA_DIR="./data"

# Required files
VOCAB_PATH="$DATA_DIR/tokenizer.vocab"
MERGES_PATH="$DATA_DIR/tokenizer.merges"
TRAIN_DATA="$DATA_DIR/train_data.npy"
VAL_DATA="$DATA_DIR/val_data.npy"

# Check required files
for file in "$VOCAB_PATH" "$MERGES_PATH" "$TRAIN_DATA"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

# Log file
LOG_FILE="$EXPERIMENT_DIR/logs/resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$EXPERIMENT_DIR/logs"

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
echo "Additional iterations: $ADDITIONAL_ITERS"
echo "Log file: $LOG_FILE"
echo ""

# Validation data parameter
VAL_DATA_PARAM=""
if [ -f "$VAL_DATA" ]; then
    VAL_DATA_PARAM="--val_data $VAL_DATA"
fi

# Resume training - using medium configuration as base
python -m cs336_basics.train \
    --vocab_path "$VOCAB_PATH" \
    --merges_path "$MERGES_PATH" \
    --train_data "$TRAIN_DATA" \
    $VAL_DATA_PARAM \
    --vocab_size 20000 \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --d_ff 3072 \
    --batch_size 32 \
    --context_length 1024 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_iters $ADDITIONAL_ITERS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --load_checkpoint "$CHECKPOINT_PATH" \
    --log_interval 100 \
    --eval_interval 2000 \
    --save_interval 10000 \
    --device auto \
    # 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training resumed and completed!"
echo "Log file: $LOG_FILE"
