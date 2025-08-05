#!/bin/bash

# 文本生成脚本
# 使用方法: ./generate_text.sh <checkpoint_path> [prompt] [config]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [prompt] [config]"
    echo "Example: $0 ./experiments/checkpoints/checkpoint_5000.pt \"Once upon a time\" small"
    echo "Configs: debug, small (default: small)"
    exit 1
fi

CHECKPOINT_PATH="$1"
PROMPT=${2:-"One day, a curious cat was"}
CONFIG=${3:-"small"}

# Default data directory
DATA_DIR="/root/autodl-tmp/data"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Common parameters
VOCAB_PATH="$DATA_DIR/tokenizer.vocab"
MERGES_PATH="$DATA_DIR/tokenizer.merges"

# Check if tokenizer files exist
if [ ! -f "$VOCAB_PATH" ]; then
    echo "Error: Vocabulary file not found: $VOCAB_PATH"
    exit 1
fi

if [ ! -f "$MERGES_PATH" ]; then
    echo "Error: Merges file not found: $MERGES_PATH"
    exit 1
fi

# Configuration-specific parameters
case $CONFIG in
    "debug")
        echo "Using DEBUG configuration..."
        VOCAB_SIZE=10000
        D_MODEL=512
        N_LAYERS=4
        N_HEADS=16
        D_FF=1344
        CONTEXT_LENGTH=256
        MAX_NEW_TOKENS=50
        TEMPERATURE=0.8
        TOP_P=0.9
        ;;
    "small")
        echo "Using SMALL configuration..."
        VOCAB_SIZE=10000
        D_MODEL=512
        N_LAYERS=4
        N_HEADS=16
        D_FF=1344
        CONTEXT_LENGTH=256
        MAX_NEW_TOKENS=100
        TEMPERATURE=0.8
        TOP_P=0.9
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Available configurations: debug, small"
        exit 1
        ;;
esac

echo "Generating text from checkpoint: $CHECKPOINT_PATH"
echo "Prompt: \"$PROMPT\""
echo "Model config: $CONFIG"
echo "Parameters: d_model=$D_MODEL, n_layers=$N_LAYERS, n_heads=$N_HEADS"
echo "Generation: max_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, top_p=$TOP_P"
echo ""

# Run text generation
python -m cs336_basics.decoding \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --vocab_path "$VOCAB_PATH" \
    --merges_path "$MERGES_PATH" \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --d_ff $D_FF \
    --context_length $CONTEXT_LENGTH \
    --prompt "$PROMPT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --device auto
