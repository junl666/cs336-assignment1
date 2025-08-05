# 训练脚本快速使用指南

## 🚀 快速开始

### 方法1: 一键完整流水线（推荐新手）
```bash
# 从原始文本文件开始，完成整个流水线
./prepare_and_train.sh corpus.txt small
```

### 方法2: 分步执行（推荐进阶用户）
```bash
# 1. 训练tokenizer
python -m cs336_basics.train_tokenizer \
    --input_path corpus.txt \
    --vocab_size 20000 \
    --output_path data/tokenizer.vocab

# 2. 转换数据为tokens
python -m cs336_basics.data2token \
    --data_path corpus.txt \
    --vocab_path data/tokenizer.vocab \
    --merges_path data/tokenizer.merges \
    --output_path data/train_data.npy \
    --method streaming

# 3. 开始训练
./run_training.sh small ./data ./experiments
```

## 📋 可用配置

| 配置 | 模型大小 | 参数量 | 推荐用途 |
|------|----------|--------|----------|
| `debug` | 极小 | ~0.5M | 快速测试、调试 |
| `small` | 小 | ~50M | 实验、学习 |
| `medium` | 中等 | ~117M | 小规模应用 |
| `large` | 大 | ~350M | 生产环境 |

## 🔄 恢复训练

```bash
# 从检查点恢复训练
./resume_training.sh ./experiments/checkpoints/checkpoint_50000.pt 100000
```

## 📊 监控训练

训练日志保存在 `./experiments/logs/` 目录中，包含：
- 训练损失
- 验证损失（如果有验证数据）
- 训练速度
- 检查点保存信息

## 🛠️ 故障排除

### 内存不足
```bash
# 使用更小的配置
./run_training.sh debug

# 或者自定义小批次大小（编辑脚本中的BATCH_SIZE）
```

### 训练太慢
```bash
# 检查GPU是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 使用更大的批次大小（如果内存允许）
```

### 数据问题
```bash
# 检查tokenizer文件
ls -la data/tokenizer.*

# 检查token数据
python -c "import numpy as np; print(np.load('data/train_data.npy').shape)"
```

## 📁 文件结构

训练完成后的目录结构：
```
./
├── data/
│   ├── tokenizer.vocab      # 词汇表
│   ├── tokenizer.merges     # 合并规则
│   ├── train_data.npy       # 训练数据tokens
│   └── val_data.npy         # 验证数据tokens
├── experiments/
│   ├── checkpoints/         # 模型检查点
│   └── logs/               # 训练日志
└── run_training.sh         # 训练脚本
```

## 🎯 下一步

训练完成后，你可以：
1. 使用检查点进行文本生成
2. 在新数据上微调模型
3. 评估模型性能
4. 部署模型到生产环境
