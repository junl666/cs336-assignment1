#!/bin/bash

# 语言模型训练脚本
# 使用方法: ./run_training.sh [config] [data_dir] [output_dir]

set -e  # Exit on any error

# Default configuration
CONFIG=${1:-"small"}
DATA_DIR=${2:-"./data"}
OUTPUT_DIR=${3:-"./experiments"}

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"

# Common parameters
VOCAB_PATH="$DATA_DIR/tokenizer.vocab"
MERGES_PATH="$DATA_DIR/tokenizer.merges"
TRAIN_DATA="$DATA_DIR/train_data.npy"
VAL_DATA="$DATA_DIR/val_data.npy"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

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
    "debug"|"tiny")
        echo "Using DEBUG/TINY configuration for quick testing..."
        VOCAB_SIZE=10000
        D_MODEL=128
        N_LAYERS=2
        N_HEADS=2
        D_FF=512
        BATCH_SIZE=8
        CONTEXT_LENGTH=128
        MAX_ITERS=1000
        LOG_INTERVAL=50
        EVAL_INTERVAL=200
        SAVE_INTERVAL=500
        ;;
    "small")
        echo "Using SMALL configuration..."
        VOCAB_SIZE=10000
        D_MODEL=512
        N_LAYERS=8
        N_HEADS=8
        D_FF=2048
        BATCH_SIZE=16
        CONTEXT_LENGTH=512
        MAX_ITERS=50000
        LOG_INTERVAL=100
        EVAL_INTERVAL=1000
        SAVE_INTERVAL=5000
        ;;
    "medium")
        echo "Using MEDIUM configuration..."
        VOCAB_SIZE=20000
        D_MODEL=768
        N_LAYERS=12
        N_HEADS=12
        D_FF=3072
        BATCH_SIZE=32
        CONTEXT_LENGTH=1024
        MAX_ITERS=100000
        LOG_INTERVAL=100
        EVAL_INTERVAL=2000
        SAVE_INTERVAL=10000
        ;;
    "large")
        echo "Using LARGE configuration..."
        VOCAB_SIZE=50000
        D_MODEL=1024
        N_LAYERS=24
        N_HEADS=16
        D_FF=4096
        BATCH_SIZE=64
        CONTEXT_LENGTH=2048
        MAX_ITERS=200000
        LOG_INTERVAL=100
        EVAL_INTERVAL=5000
        SAVE_INTERVAL=20000
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Available configurations: debug, tiny, small, medium, large"
        exit 1
        ;;
esac

# Log file
LOG_FILE="$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with configuration: $CONFIG"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Model parameters: d_model=$D_MODEL, n_layers=$N_LAYERS, n_heads=$N_HEADS"
echo "Training parameters: batch_size=$BATCH_SIZE, context_length=$CONTEXT_LENGTH, max_iters=$MAX_ITERS"
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
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_iters $MAX_ITERS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_interval $LOG_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --device auto \
    # 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "Log file: $LOG_FILE"
