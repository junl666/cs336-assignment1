from einops import repeat, einsum
import torch


t = torch.tensor([[1, 2, 3],
                 [4, 5, 6],]) # shape (2, 3)

# repeat the tensor along the first dimension twice
t_repeated = repeat(t, 'b d -> (b r) d', r=2)
# t_repeated will have shape (4, 3)
print(t_repeated)

print("-"*20)
# 测试不同的 einsum 方式
t = torch.tensor([[1, 2],
                 [4, 5],]) # shape (2, 2)

a = torch.tensor([1, 2])

print(einsum(t, a, "d d, d -> d")) # 输出: tensor([ 1, 10])
print(einsum(t, a, "i j, j -> i"))  # 输出: tensor([ 5, 14])
print(t @ a)  # 输出: tensor([ 5, 14])