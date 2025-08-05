#!/bin/bash

# 完整的数据准备和训练流水线
# 使用方法: ./prepare_and_train.sh <raw_text_file> [config]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <raw_text_file> [config]"
    echo "Configurations: debug, small, medium, large"
    echo "Example: $0 corpus.txt small"
    exit 1
fi

RAW_TEXT="$1"
CONFIG=${2:-"small"}

if [ ! -f "$RAW_TEXT" ]; then
    echo "Error: Raw text file not found: $RAW_TEXT"
    exit 1
fi

echo "=== 完整训练流水线 ==="
echo "原始文本文件: $RAW_TEXT"
echo "配置: $CONFIG"
echo ""

# Create directories
mkdir -p data
mkdir -p experiments/logs

# Step 1: Train tokenizer
echo "步骤 1/4: 训练BPE tokenizer..."
python -m cs336_basics.train_tokenizer \
    --input_path "$RAW_TEXT" \
    --vocab_size 20000 \
    --output_path data/tokenizer.vocab \
    --special_tokens "<|endoftext|>"

if [ $? -ne 0 ]; then
    echo "Error: Tokenizer training failed"
    exit 1
fi

echo "Tokenizer训练完成!"
echo ""

# Step 2: Tokenize data
echo "步骤 2/4: 将文本数据转换为tokens..."

# Split data into train/val (90%/10%)
TOTAL_LINES=$(wc -l < "$RAW_TEXT")
TRAIN_LINES=$((TOTAL_LINES * 9 / 10))

head -n $TRAIN_LINES "$RAW_TEXT" > data/train_text.txt
tail -n +$((TRAIN_LINES + 1)) "$RAW_TEXT" > data/val_text.txt

echo "数据分割完成: 训练集 $TRAIN_LINES 行, 验证集 $((TOTAL_LINES - TRAIN_LINES)) 行"

# Tokenize training data
python -m cs336_basics.data2token \
    --data_path data/train_text.txt \
    --vocab_path data/tokenizer.vocab \
    --merges_path data/tokenizer.merges \
    --output_path data/train_data.npy \
    --method streaming

if [ $? -ne 0 ]; then
    echo "Error: Training data tokenization failed"
    exit 1
fi

# Tokenize validation data
python -m cs336_basics.data2token \
    --data_path data/val_text.txt \
    --vocab_path data/tokenizer.vocab \
    --merges_path data/tokenizer.merges \
    --output_path data/val_data.npy \
    --method streaming

if [ $? -ne 0 ]; then
    echo "Error: Validation data tokenization failed"
    exit 1
fi

echo "数据tokenization完成!"
echo ""

# Step 3: Display data statistics
echo "步骤 3/4: 数据统计..."
TRAIN_TOKENS=$(python -c "import numpy as np; data = np.load('data/train_data.npy'); print(len(data))")
VAL_TOKENS=$(python -c "import numpy as np; data = np.load('data/val_data.npy'); print(len(data))")

echo "训练数据: $TRAIN_TOKENS tokens"
echo "验证数据: $VAL_TOKENS tokens"
echo "总计: $((TRAIN_TOKENS + VAL_TOKENS)) tokens"
echo ""

# Step 4: Start training
echo "步骤 4/4: 开始训练..."
./run_training.sh "$CONFIG" "./data" "./experiments"

echo ""
echo "=== 流水线完成! ==="
echo "检查点保存在: ./experiments/checkpoints/"
echo "日志保存在: ./experiments/logs/"
echo ""
echo "要恢复训练，使用:"
echo "./resume_training.sh ./experiments/checkpoints/checkpoint_XXXXX.pt"
