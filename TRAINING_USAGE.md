# 训练脚本使用指南

## 完整训练脚本 (`cs336_basics/train.py`)

这是一个功能完整的语言模型训练脚本，集成了所有必要的组件。

## 便捷训练脚本

### 快速开始训练
使用预配置的训练脚本：

```bash
# 小规模调试训练，使用20%的数据进行快速测试
./run_training.sh debug

# 小规模完整训练
./run_training.sh small

# 指定数据目录和输出目录
./run_training.sh small /path/to/data ./my_experiment
```

### 恢复训练
从检查点继续训练：

```bash
# 基本恢复
./resume_training.sh ./experiments/checkpoints/checkpoint_5000.pt

# 指定额外训练步数
./resume_training.sh ./experiments/checkpoints/checkpoint_5000.pt 20000

# 指定数据目录
./resume_training.sh ./experiments/checkpoints/checkpoint_5000.pt 20000 /path/to/data
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
- `--learning_rate`: 学习率 (已废弃，使用max_lr)
- `--max_lr`: 最大学习率 (默认: 3e-4)
- `--min_lr`: 最小学习率 (默认: 3e-5)
- `--warmup_steps`: 预热步数 (默认: 1000)
- `--decay_steps`: 衰减步数 (默认: 50000)
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
- `--log_dir`: TensorBoard日志目录 (默认: ./logs)
- `--disable_tensorboard`: 禁用TensorBoard记录
- `--device`: 设备选择 (默认: auto, 可选: cpu, cuda, cuda:0等)

## 主要特性

### 1. 内存高效的数据加载
- 使用 `np.memmap` 处理大型数据集
- 随机批次采样，避免内存溢出

### 2. 智能检查点系统
- 自动保存模型状态、优化器状态和训练进度
- 支持训练中断后的无缝恢复
- 定期和最终检查点保存

### 3. 学习率调度
- **预热阶段**: 从0线性增长到max_lr
- **余弦衰减**: 从max_lr余弦衰减到min_lr
- **最小学习率保持**: 衰减完成后保持min_lr
- 自动防止训练初期的梯度爆炸

### 4. TensorBoard集成
- 实时训练和验证损失可视化
- 学习率变化曲线
- 训练性能指标监控
- 自动创建带时间戳的日志目录

### 5. 梯度裁剪
- 防止梯度爆炸问题
- 可配置的梯度范数阈值

### 6. 完整的训练监控
- 实时训练损失显示
- 定期验证损失评估
- 训练时间和学习率跟踪
- 进度条和时间估算

### 7. 灵活的模型配置
- 支持不同规模的Transformer模型
- RoPE位置编码
- 可配置的前馈层维度

### 8. 便捷脚本
- 预配置的训练配置（debug、small）
- 一键训练和恢复脚本
- 自动化的目录管理

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

## TensorBoard监控

### 启动TensorBoard
训练开始后，在另一个终端中运行：

```bash
tensorboard --logdir ./experiments/logs
```

然后在浏览器中访问 `http://localhost:6006` 查看：
- 训练和验证损失曲线
- 学习率变化
- 训练性能指标

### 查看指标
TensorBoard中可以看到：
- `Loss/Train`: 训练损失
- `Loss/Validation`: 验证损失  
- `Learning_Rate`: 学习率变化
- `Time_per_Batch`: 每批次处理时间

## 训练恢复

从检查点恢复训练：

```bash
python -m cs336_basics.train \
    --vocab_path tokenizer.vocab \
    --merges_path tokenizer.merges \
    --train_data train_data.npy \
    --load_checkpoint ./checkpoints/checkpoint_50000.pt \
    --max_iters 50000  # 额外训练50000步
```

**注意**: `max_iters`参数在恢复训练时表示额外的训练步数，不是总步数。

## 配置说明

### debug配置
- 使用20%的数据进行快速测试
- 小模型参数 (d_model=512, n_layers=4)
- 短训练时间 (2000步)
- 适合功能测试和调试

### small配置  
- 使用全部数据
- 小模型参数 (d_model=512, n_layers=4)
- 中等训练时间 (10000步)
- 适合初步实验

## 训练监控输出

训练过程中会看到类似输出：

```
Using device: cuda
TensorBoard logging to: ./experiments/logs/run_20250805_143022
To view logs, run: tensorboard --logdir ./experiments/logs
Loading tokenizer...
Tokenizer loaded.
Loading training data...
Loading validation data...
Creating model...
Model created with 117,440,512 total parameters (117,440,512 trainable)
Learning rate schedule: max_lr=0.0003, min_lr=3e-05, warmup_steps=1000, decay_steps=50000
Starting training...
Iter    100 | Loss: 8.2341 | Time: 0.145s | LR: 3.00e-05
Iter    200 | Loss: 7.8923 | Time: 0.142s | LR: 6.00e-05
Iter   1000 | Loss: 6.4521 | Time: 0.140s | LR: 3.00e-04
Evaluating...
Iter   1000 | Val Loss: 6.1234
Saving checkpoint to ./checkpoints/checkpoint_5000.pt
```

学习率会根据调度策略自动变化：
- 步骤0-1000：线性预热从0到max_lr
- 步骤1000-50000：余弦衰减从max_lr到min_lr  
- 步骤50000+：保持min_lr

## 故障排除

### 内存不足
- 减少 `--batch_size`
- 减少 `--context_length`
- 使用更小的模型维度
- 使用debug配置进行测试

### 训练不稳定
- 降低 `--max_lr`
- 增加 `--warmup_steps`
- 减少 `--max_grad_norm`
- 增加 `--weight_decay`

### 训练太慢
- 增加 `--batch_size`（如果内存允许）
- 减少 `--eval_interval` 和 `--save_interval`
- 使用更快的设备
- 禁用TensorBoard (`--disable_tensorboard`)

### 学习率调度问题
- 确保 `warmup_steps < decay_steps`
- 调整 `max_lr` 和 `min_lr` 的比例
- 根据总训练步数调整 `decay_steps`

### TensorBoard无法启动
- 检查端口6006是否被占用
- 确保tensorboard已安装: `pip install tensorboard`
- 使用不同端口: `tensorboard --logdir ./logs --port 6007`

## 最佳实践

### 1. 数据准备
- 使用`data2token_streaming`处理大文件
- 确保验证集与训练集分离
- 定期备份tokenizer文件

### 2. 训练策略
- 先用debug配置测试流程
- 使用TensorBoard监控训练进度
- 定期保存检查点（建议每1000-5000步）
- 从小模型开始，逐步增大

### 3. 超参数调优
- 学习率是最重要的超参数
- 预热步数通常设为总步数的5-10%
- 批次大小越大越稳定，但需要更多内存
- 梯度裁剪范数建议在0.5-2.0之间

### 4. 资源管理
- 使用便捷脚本自动化训练流程
- 定期清理旧的检查点文件
- 监控磁盘空间使用情况
