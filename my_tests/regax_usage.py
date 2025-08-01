import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = "Hello Hello <|endoftext|>World [MASK] Python\\nCode"

special_tokens = ["<|endoftext|>", "[MASK]", "\\n"]
# 转义并构建分割模式
escaped_tokens = [re.escape(token) for token in special_tokens]
split_pattern = "|".join(escaped_tokens)

chunks = re.split('(' + split_pattern + ')', text)

print("Chunks:", chunks)

# token = "<|endoftext|>"
# encoded = tuple(token.encode('utf-8'))
# print("Encoded token:", encoded)

# token2 = "H"
# encoded2 = tuple(token2.encode('utf-8'))
# print("Encoded token2:", encoded2)

# rt = re.findall(PAT, text)

# print(rt)

# rt2 = re.finditer(PAT, "some text that i'll pre-tokenize")

# print([m.group() for m in rt2])