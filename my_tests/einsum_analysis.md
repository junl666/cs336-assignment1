# Einsum 矩阵乘法陷阱（珍爱生命，谨慎使用einsum）

## 问题

实现 RoPE 时，以下 einsum 形状正确但结果错误：

```python
# 错误 - 不是矩阵乘法
einsum(x, rope_values, "... seq d_k, ... seq d_k d_k -> ... seq d_k")
```

## 原因

不同的索引模式执行不同的数学操作：

```python
t = torch.tensor([[1, 2], [4, 5]])
a = torch.tensor([1, 2])

einsum(t, a, "d d, d -> d")    # [1*1+2*2, 4*1+5*2] = [5, 14] ✗ 错误
einsum(t, a, "i j, j -> i")    # [1*1+2*2, 4*1+5*2] = [5, 14] ✓ 正确
t @ a                          # [1*1+2*2, 4*1+5*2] = [5, 14] ✓ 参考
```

关键：`"d d, d -> d"` 表示对角线操作，不是矩阵乘法。

## 解决方案

```python
# 方案1: 正确的 einops.einsum
einsum(rope_values, x, "... seq i j, ... seq j -> ... seq i")

# 方案2: torch.einsum  
torch.einsum('...ij,...j->...i', rope_values, x)

# 方案3: 直接矩阵乘法
torch.matmul(rope_values, x.unsqueeze(-1)).squeeze(-1)
```

## 教训

1. 索引名称要反映实际的数学操作
2. 形状匹配 ≠ 操作正确
3. 优先用 `torch.matmul`，再考虑 einsum 优化
