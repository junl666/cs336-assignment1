# 训练脚本使用指南

## 完整训练脚本 (`cs336_basics/train.py`)

这是一个功能完整的语言模型训练脚本，集成了所有必要的组件。

### 基本使用

```bash
python -m cs336_basics.train \
    --vocab_path tokenizer.vocab \
    --merges_path tokenizer.merges \
    --train_data train_data.npy \
    --val_data val_data.npy \
    --vocab_size 10000 \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --batch_size 32 \
    --context_length 512 \
    --learning_rate 3e-4 \
    --max_iters 100000 \
    --checkpoint_dir ./checkpoints
```

## 脚本参数详解

### 模型超参数
- `--vocab_size`: 词汇表大小 (默认: 10000)
- `--d_model`: 模型维度 (默认: 768)
- `--n_layers`: 层数 (默认: 12)
- `--n_heads`: 注意力头数 (默认: 12)
- `--max_seq_len`: 最大序列长度 (默认: 1024)
- `--rope_theta`: RoPE缩放因子 (默认: 10000.0)
- `--d_ff`: 前馈层维度 (默认: 3072)

### 训练超参数
- `--batch_size`: 批次大小 (默认: 32)
- `--context_length`: 上下文长度 (默认: 512)
- `--learning_rate`: 学习率 (默认: 3e-4)
- `--weight_decay`: 权重衰减 (默认: 0.01)
- `--beta1`: Adam优化器beta1 (默认: 0.9)
- `--beta2`: Adam优化器beta2 (默认: 0.999)
- `--max_grad_norm`: 梯度裁剪范数 (默认: 1.0)
- `--max_iters`: 最大训练迭代数 (默认: 100000)

### 数据和检查点
- `--vocab_path`: tokenizer词汇表文件路径 (必需)
- `--merges_path`: tokenizer合并规则文件路径 (必需)
- `--train_data`: 训练数据路径 (必需)
- `--val_data`: 验证数据路径 (可选)
- `--checkpoint_dir`: 检查点保存目录 (默认: ./checkpoints)
- `--load_checkpoint`: 加载检查点路径 (可选)

### 日志和评估
- `--log_interval`: 日志记录间隔 (默认: 100)
- `--eval_interval`: 评估间隔 (默认: 1000)
- `--save_interval`: 检查点保存间隔 (默认: 5000)
- `--num_eval_batches`: 评估批次数 (默认: 100)
- `--device`: 设备选择 (默认: auto, 可选: cpu, cuda, cuda:0等)

## 主要特性

### 1. 内存高效的数据加载
- 使用 `np.memmap` 处理大型数据集
- 随机批次采样，避免内存溢出

### 2. 智能检查点系统
- 自动保存模型状态、优化器状态和训练进度
- 支持训练中断后的无缝恢复
- 定期和最终检查点保存

### 3. 梯度裁剪
- 防止梯度爆炸问题
- 可配置的梯度范数阈值

### 4. 完整的训练监控
- 实时训练损失显示
- 定期验证损失评估
- 训练时间和学习率跟踪

### 5. 灵活的模型配置
- 支持不同规模的Transformer模型
- RoPE位置编码
- 可配置的前馈层维度

## 数据准备

### 支持的数据格式
1. **NumPy数组文件** (.npy) - 推荐
2. **二进制文件** (.bin)

数据应该是int32类型的token ID数组。

### 数据准备示例

```python
# 使用我们的data2token脚本
python -m cs336_basics.data2token \
    --data_path raw_text.txt \
    --vocab_path tokenizer.vocab \
    --merges_path tokenizer.merges \
    --output_path train_data.npy \
    --method streaming

# 或者手动创建
import numpy as np
tokens = np.random.randint(0, 10000, size=1000000, dtype=np.int32)
np.save('train_data.npy', tokens)
```

## 训练恢复

从检查点恢复训练：

```bash
python -m cs336_basics.train \
    --vocab_path tokenizer.vocab \
    --merges_path tokenizer.merges \
    --train_data train_data.npy \
    --load_checkpoint ./checkpoints/checkpoint_50000.pt \
    --max_iters 200000
```

## 训练监控输出

训练过程中会看到类似输出：

```
Using device: cuda
Loading tokenizer...
Tokenizer loaded.
Loading training data...
Loading validation data...
Creating model...
Model created with 117,440,512 total parameters (117,440,512 trainable)
Starting training...
Iter    100 | Loss: 8.2341 | Time: 0.145s | LR: 3.00e-04
Iter    200 | Loss: 7.8923 | Time: 0.142s | LR: 3.00e-04
Evaluating...
Iter   1000 | Val Loss: 7.4521
Saving checkpoint to ./checkpoints/checkpoint_5000.pt
```

## 最佳实践

### 小模型（调试用）
```bash
python -m cs336_basics.train \
    --vocab_size 1000 \
    --d_model 256 \
    --n_layers 4 \
    --n_heads 4 \
    --batch_size 16 \
    --max_iters 10000
```

### 中等模型
```bash
python -m cs336_basics.train \
    --vocab_size 10000 \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --batch_size 32 \
    --max_iters 100000
```

### 大模型
```bash
python -m cs336_basics.train \
    --vocab_size 50000 \
    --d_model 1024 \
    --n_layers 24 \
    --n_heads 16 \
    --batch_size 64 \
    --max_iters 500000
```

## 故障排除

### 内存不足
- 减少 `--batch_size`
- 减少 `--context_length`
- 使用更小的模型维度

### 训练不稳定
- 降低 `--learning_rate`
- 减少 `--max_grad_norm`
- 增加 `--weight_decay`

### 训练太慢
- 增加 `--batch_size`（如果内存允许）
- 减少 `--eval_interval` 和 `--save_interval`
- 使用更快的设备
