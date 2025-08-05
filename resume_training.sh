#!/bin/bash

# 从检查点恢复训练脚本
# 使用方法: ./resume_training.sh <checkpoint_path> [additional_iters] [data_dir]

set -e  # Exit on any error

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [additional_iters] [data_dir]"
    echo "Example: $0 ./experiments/checkpoints/checkpoint_10000.pt 50000 /root/autodl-tmp/data"
    exit 1
fi

CHECKPOINT_PATH="$1"
ADDITIONAL_ITERS=${2:-50000}
DATA_DIR=${3:-"/root/autodl-tmp/data"}

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Extract directory information from checkpoint path
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
EXPERIMENT_DIR=$(dirname "$CHECKPOINT_DIR")

# Create logs directory if it doesn't exist
mkdir -p "$EXPERIMENT_DIR/logs"

# Common parameters
VOCAB_PATH="$DATA_DIR/tokenizer.vocab"
MERGES_PATH="$DATA_DIR/tokenizer.merges"
TRAIN_DATA="$DATA_DIR/train_data.npy"
VAL_DATA="$DATA_DIR/valid_data.npy"
LOG_DIR="$EXPERIMENT_DIR/logs"

# Check required files
if [ ! -f "$VOCAB_PATH" ]; then
    echo "Error: Vocabulary file not found: $VOCAB_PATH"
    echo "Please run tokenizer training first or provide correct data directory."
    exit 1
fi

if [ ! -f "$MERGES_PATH" ]; then
    echo "Error: Merges file not found: $MERGES_PATH"
    echo "Please run tokenizer training first or provide correct data directory."
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Please run data tokenization first or provide correct data directory."
    exit 1
fi

# Default resume configuration
VOCAB_SIZE=10000
D_MODEL=512
N_LAYERS=4
N_HEADS=16
D_FF=1344
BATCH_SIZE=16
CONTEXT_LENGTH=256
MAX_LR=3e-4
MIN_LR=3e-5
WARMUP_STEPS=1000
DECAY_STEPS=8000
LOG_INTERVAL=100
EVAL_INTERVAL=500
SAVE_INTERVAL=500

# Log file
LOG_FILE="$EXPERIMENT_DIR/logs/resume_$(date +%Y%m%d_%H%M%S).log"

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
echo "Additional iterations: $ADDITIONAL_ITERS"
echo "Data directory: $DATA_DIR"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Log file: $LOG_FILE"
echo "TensorBoard logs: $LOG_DIR"
echo "Model parameters: d_model=$D_MODEL, n_layers=$N_LAYERS, n_heads=$N_HEADS"
echo "Training parameters: batch_size=$BATCH_SIZE, context_length=$CONTEXT_LENGTH, max_iters(additional)=$ADDITIONAL_ITERS"
echo "Learning rate schedule: max_lr=$MAX_LR, min_lr=$MIN_LR, warmup_steps=$WARMUP_STEPS, decay_steps=$DECAY_STEPS"
echo ""
echo "To monitor training progress with TensorBoard, run:"
echo "  tensorboard --logdir $LOG_DIR"
echo ""

# Validation data parameter
VAL_DATA_PARAM=""
if [ -f "$VAL_DATA" ]; then
    VAL_DATA_PARAM="--val_data $VAL_DATA"
    echo "Using validation data: $VAL_DATA"
else
    echo "Warning: Validation data not found: $VAL_DATA"
    echo "Training without validation."
fi

# Resume training
python -m cs336_basics.train \
    --vocab_path "$VOCAB_PATH" \
    --merges_path "$MERGES_PATH" \
    --train_data "$TRAIN_DATA" \
    $VAL_DATA_PARAM \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --context_length $CONTEXT_LENGTH \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --decay_steps $DECAY_STEPS \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_iters $ADDITIONAL_ITERS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --load_checkpoint "$CHECKPOINT_PATH" \
    --log_dir "$LOG_DIR" \
    --log_interval $LOG_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --device auto \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training resumed and completed!"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "TensorBoard logs saved in: $LOG_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "To view training metrics, run:"
echo "  tensorboard --logdir $LOG_DIR"
