#!/bin/bash

# 语言模型训练脚本
# 使用方法: ./run_training.sh [config] [data_dir] [output_dir]

set -e  # Exit on any error

# Default configuration
CONFIG=${1:-"small"}
DATA_DIR=${2:-"/root/autodl-tmp/data"}
OUTPUT_DIR=${3:-"./experiments"}
DEBUG_RATIO=1.0  # Default to full data

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"

# Common parameters
VOCAB_PATH="$DATA_DIR/tokenizer.vocab"
MERGES_PATH="$DATA_DIR/tokenizer.merges"
TRAIN_DATA="$DATA_DIR/train_data.npy"
VAL_DATA="$DATA_DIR/valid_data.npy"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"

# Check if required files exist
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

# Configuration-specific parameters
case $CONFIG in
    "debug")
        echo "Using DEBUG configuration for quick testing on a subset of the dataset."
        DEBUG_RATIO=0.2     # Use 20% of the data
        VOCAB_SIZE=10000
        D_MODEL=512
        N_LAYERS=4
        N_HEADS=16
        D_FF=1344
        BATCH_SIZE=16
        CONTEXT_LENGTH=256
        MAX_ITERS=2000
        MAX_LR=3e-4
        MIN_LR=3e-5
        WARMUP_STEPS=200
        DECAY_STEPS=1500
        LOG_INTERVAL=100
        EVAL_INTERVAL=200
        SAVE_INTERVAL=500
        ;;
    "small")
        echo "Using SMALL configuration on the full dataset..."
        VOCAB_SIZE=10000
        D_MODEL=512
        N_LAYERS=4
        N_HEADS=16
        D_FF=1344
        BATCH_SIZE=16
        CONTEXT_LENGTH=256
        MAX_ITERS=10000
        MAX_LR=3e-4
        MIN_LR=3e-5
        WARMUP_STEPS=1000
        DECAY_STEPS=8000
        LOG_INTERVAL=100
        EVAL_INTERVAL=500
        SAVE_INTERVAL=500
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Available configurations: debug, small"
        exit 1
        ;;
esac

# Log file
LOG_FILE="$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with configuration: $CONFIG"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "TensorBoard logs: $LOG_DIR"
echo "Model parameters: d_model=$D_MODEL, n_layers=$N_LAYERS, n_heads=$N_HEADS"
echo "Training parameters: batch_size=$BATCH_SIZE, context_length=$CONTEXT_LENGTH, max_iters=$MAX_ITERS"
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

# Run training
python -m cs336_basics.train \
    --debug_ratio $DEBUG_RATIO \
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
    --max_iters $MAX_ITERS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --log_interval $LOG_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --device auto \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "TensorBoard logs saved in: $LOG_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "To view training metrics, run:"
echo "  tensorboard --logdir $LOG_DIR"
