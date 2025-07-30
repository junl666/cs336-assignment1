import os
from typing import BinaryIO, Union, DefaultDict
import regex as re
import multiprocessing as mp
from functools import partial



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def preprocess_for_bpe(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Properly preprocess text by splitting on special tokens.
    """    
    # 转义并构建分割模式
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    
    # 分割并清理
    chunks = re.split(split_pattern, text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    word_count_dict = dict[str, int]()  # 使用字典来存储单词计数

    for chunk in chunks:
        tokens = re.findall(PAT, chunk)
        for token_text in tokens:
            if token_text:  # 只检查是否为空，不strip
                word_count_dict[token_text] = word_count_dict.get(token_text, 0) + 1
    
    byte_word_count_dict = {}
    for word, count in word_count_dict.items():
        # 将每个字节转换为字节字面量
        byte_word = tuple(bytes([b]) for b in word.encode('utf-8'))
        if byte_word in byte_word_count_dict:
            byte_word_count_dict[byte_word] += count
        else:
            byte_word_count_dict[byte_word] = count 
    return byte_word_count_dict


def merge(byte_word_count_dict: dict[tuple[bytes, ...], int], num_merge: int) -> list[tuple[bytes, bytes]]:
    from dataclasses import dataclass
    @dataclass
    class WordUtil:
        word_bytes: tuple[bytes, ...]
        word_count: int
        valid_subbytes_index_list: list[Union[int, tuple[int, int]]] # 
        bytes_pair_cnt: dict[tuple[bytes, bytes], int] # consecutive bytes pair count

        def update_bytes_pair_cnt(self):
            """
            Update the bytes_pair_cnt for this word according to the valid_subbytes_index_list.
            For example, if the word is (b'H', b'e', b'l', b'l', b'o'), 
            valid_subbytes_index_list = [0, 1, 2, 3, 4] means all bytes are valid.
            Then the bytes_pair_cnt will be:
            {
                (b'H', b'e'): 1 * word_count,
                (b'e', b'l'): 1 * word_count,
                (b'l', b'l'): 1 * word_count,
                (b'l', b'o'): 1 * word_count,
            }
            Another example, if the word is (b'H', b'e', b'l', b'l', b'o') and
            valid_subbytes_index_list = [0, 1, 2, (3, 4)] means the last two bytes are merged.
            Then the bytes_pair_cnt will be:
            {
                (b'H', b'e'): 1 * word_count,
                (b'e', b'l'): 1 * word_count,
                (b'l', b'lo'): 1 * word_count,
            }
            """
            self.bytes_pair_cnt.clear()
            for i in range(len(self.valid_subbytes_index_list) - 1):
                current_item = self.valid_subbytes_index_list[i]
                next_item = self.valid_subbytes_index_list[i + 1]
                
                # Get the actual bytes for current position
                if isinstance(current_item, int):
                    current_bytes = self.word_bytes[current_item]
                else:  # tuple case - merged bytes
                    start, end = current_item
                    current_bytes = b''.join(self.word_bytes[start:end+1])
                
                # Get the actual bytes for next position
                if isinstance(next_item, int):
                    next_bytes = self.word_bytes[next_item]
                else:  # tuple case - merged bytes
                    start, end = next_item
                    next_bytes = b''.join(self.word_bytes[start:end+1])
                
                pair = (current_bytes, next_bytes)
                if pair in self.bytes_pair_cnt:
                    self.bytes_pair_cnt[pair] += self.word_count
                else:
                    self.bytes_pair_cnt[pair] = self.word_count

    indi_pair_cnt: list[WordUtil] = list()  # 以每个单词为单位
    # 所有bytes的计数，即是把上面的计数按key做合并, list[int]代表这个bytes出现在indi_pair_cnt的位置索引
    overall_pair_cnt: dict[tuple[bytes, bytes], list[int, list[int]]] = dict()  # key: (bytes, bytes), value: (count, [index_list])

    def update_indi_pair_cnt(index: int=None, merge_pair: tuple[bytes, bytes]= None):
        if indi_pair_cnt == []:  # initialization
            for word_bytes, word_count in byte_word_count_dict.items():
                valid_subbytes_index_list = list(range(len(word_bytes)))
                word_util = WordUtil(word_bytes, word_count, valid_subbytes_index_list, {})
                word_util.update_bytes_pair_cnt()
                indi_pair_cnt.append(word_util)
        else:
            # 辅助函数
            def get_bytes(word_bytes, item):
                if isinstance(item, int):
                    return word_bytes[item]
                start, end = item
                return b''.join(word_bytes[start:end+1])

            def merge_items(current, next_):
                if isinstance(current, int) and isinstance(next_, int):
                    return (current, next_)
                elif isinstance(current, int) and isinstance(next_, tuple):
                    return (current, next_[1])
                elif isinstance(current, tuple) and isinstance(next_, int):
                    return (current[0], next_)
                else:
                    return (current[0], next_[1])
                
            word_util = indi_pair_cnt[index]
            i = 0
            while i < len(word_util.valid_subbytes_index_list) - 1:
                current_item = word_util.valid_subbytes_index_list[i]
                next_item = word_util.valid_subbytes_index_list[i + 1]
                
                # 获取当前和下一个字节
                current_bytes = get_bytes(word_util.word_bytes, current_item)
                next_bytes = get_bytes(word_util.word_bytes, next_item)
                pair = (current_bytes, next_bytes)

                if pair == merge_pair:
                    # 处理所有合并情况
                    new_item = merge_items(current_item, next_item)
                    word_util.valid_subbytes_index_list[i] = new_item
                    word_util.valid_subbytes_index_list.pop(i + 1)
                    
                    # 不增加i，因为下一个元素现在位于当前i位置
                    continue
                i += 1
            
            word_util.update_bytes_pair_cnt()

    def update_overall_pair_cnt(word_util: WordUtil, update_type: str = "add", i: int = None):
        """
        Update the overall_pair_cnt based on the word_util's bytes_pair_cnt.
        If update_type is "add", it adds the counts and indices.
        If update_type is "subtract", it subtracts the counts and removes the indices.
        """
        if (update_type == "add"):
            for pair, count in word_util.bytes_pair_cnt.items():
                if pair in overall_pair_cnt:
                    overall_pair_cnt[pair][0] += count
                    overall_pair_cnt[pair][1].append(i)
                else:
                    overall_pair_cnt[pair] = [count, [i]]
        elif (update_type == "subtract"):
            for pair, count in word_util.bytes_pair_cnt.items():
                if pair in overall_pair_cnt:
                    overall_pair_cnt[pair][0] -= count
                    overall_pair_cnt[pair][1].remove(i)
                else:
                    raise ValueError(f"Pair {pair} not found in overall_pair_cnt for subtraction")
        else:
            raise ValueError("Invalid update_type. Use 'add' or 'subtract'.")
    
    max_pair_tuple = tuple()
    max_pair_count = 0
    # Initialize indi_pair_cnt
    update_indi_pair_cnt()
    # Initialize overall_pair_cnt
    for i, word_util in enumerate(indi_pair_cnt):
        for pair, count in word_util.bytes_pair_cnt.items():
            if pair in overall_pair_cnt:
                overall_pair_cnt[pair][0] += count
                overall_pair_cnt[pair][1].append(i)
            else:
                overall_pair_cnt[pair] = [count, [i]]

            # Update max_pair_tuple and max_pair_count
            if overall_pair_cnt[pair][0] > max_pair_count:
                max_pair_count = overall_pair_cnt[pair][0]
                max_pair_tuple = pair
            elif overall_pair_cnt[pair][0] == max_pair_count:
                # deterministically break ties in pair frequency by preferring 
                # the lexicographically greater pair.
                if pair > max_pair_tuple:
                    max_pair_tuple = pair
                
    # Now we have the most frequent pair in max_pair_tuple
    # and its count in max_pair_count
    if max_pair_count == 0:
        raise ValueError("No pairs found to merge")
    result_tuple_list = []
    result_tuple_list.append(max_pair_tuple)

    # Iterate until num_merge
    for _ in range(num_merge - 1):
        for i in overall_pair_cnt[max_pair_tuple][1].copy():
            update_overall_pair_cnt(indi_pair_cnt[i], "subtract", i)
            update_indi_pair_cnt(i, max_pair_tuple)
            update_overall_pair_cnt(indi_pair_cnt[i], "add", i)
        # find the next max pair
        # TODO: Make it more efficient by using a priority queue
        max_pair_count = 0
        max_pair_tuple = tuple()
        for pair, (count, indices) in overall_pair_cnt.items():
            if count > max_pair_count:
                max_pair_count = count
                max_pair_tuple = pair
            elif count == max_pair_count:
                # deterministically break ties in pair frequency by preferring 
                # the lexicographically greater pair.
                if pair > max_pair_tuple:
                    max_pair_tuple = pair
        result_tuple_list.append(max_pair_tuple)
    return result_tuple_list


def process_chunk(start_end_pair: tuple[int, int], input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Process a single chunk of the file and return word counts.
    This function will be called by each process.
    """
    start, end = start_end_pair
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Run pre-tokenization on the chunk
        chunk_word_counts = preprocess_for_bpe(chunk, special_tokens)
        
    return chunk_word_counts


def merge_word_count_dicts(dict_list: list[dict[tuple[bytes, ...], int]]) -> dict[tuple[bytes, ...], int]:
    """
    Merge multiple word count dictionaries into one.
    """
    merged_dict = {}
    
    for word_count_dict in dict_list:
        for word, count in word_count_dict.items():
            if word in merged_dict:
                merged_dict[word] += count
            else:
                merged_dict[word] = count
                
    return merged_dict


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE on the input file and return the merge operations.
    """
    print("Starting BPE training with multiprocessing...")
    
    # Step 1: Find chunk boundaries
    with open(input_path, "rb") as f:
        num_processes = 4  # Use all available CPU cores
        print(f"Using {num_processes} processes")
        
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(f"Found {len(boundaries)-1} chunks")
        
        # Create start-end pairs for each chunk
        chunk_pairs = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    
    # Step 2: Process chunks in parallel
    print("Processing chunks in parallel...")
    
    # Create partial function with fixed arguments
    process_func = partial(process_chunk, input_path=input_path, special_tokens=special_tokens)
    
    # Use multiprocessing to process chunks
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_func, chunk_pairs)
    
    # Step 3: Merge all chunk results
    print("Merging chunk results...")
    byte_word_count_dict = merge_word_count_dicts(chunk_results)
    
    print(f"Total unique byte words: {len(byte_word_count_dict)}")
    print(f"Total word occurrences: {sum(byte_word_count_dict.values())}")
    
    # Step 4: Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}  # Initialize with single byte tokens
    
    # Add special tokens to vocabulary
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    
    # Step 5: Calculate number of merges needed
    num_merge = vocab_size - len(vocab)  # Number of merges to perform
    
    if num_merge <= 0:
        raise ValueError(f"Vocab size ({vocab_size}) must be greater than initial vocab size ({len(vocab)})")
    
    print(f"Performing {num_merge} merge operations...")
    
    # Step 6: Perform BPE merges
    merges = merge(byte_word_count_dict, num_merge)
    
    # Step 7: Update vocabulary with merged tokens
    for merge_pair in merges:
        merged_token = merge_pair[0] + merge_pair[1]
        vocab[len(vocab)] = merged_token
    
    print("BPE training completed!")
    return vocab, merges


def test_preprocess_for_bpe():
    text = "Hello Hello <|endoftext|>World [MASK] Python\\nCode"
    tokens = ["<|endoftext|>", "[MASK]", "\\n"]
    result = preprocess_for_bpe(text, tokens)
    expected = {
        (b'H', b'e', b'l', b'l', b'o'): 2,
        (b'W', b'o', b'r', b'l', b'd'): 1,
        (b'P', b'y', b't', b'h', b'o', b'n'): 1,
        (b'C', b'o', b'd', b'e'): 1,
    }
    assert result == expected, f"Expected {expected}, but got {result}"
    print("Preprocessing test passed!")

def test_merge():
    byte_word_count_dict = {
        (b'l', b'o', b'w'): 5,
        (b'l', b'o', b'w', b'e', b'r'): 2,
        (b'w', b'i', b'd', b'e', b's', b't'): 3,
        (b'n', b'e', b'w', b'e', b's', b't'): 6,
    }

    # for num_merge = 1 
    result = merge(byte_word_count_dict, 1)
    expected = [(b's', b't')]
    assert result == expected, f"num_merge = 1: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 1!")

    # for num_merge = 2
    result = merge(byte_word_count_dict, 2)
    expected = [(b's', b't'), (b'e', b'st')]
    assert result == expected, f"num_merge = 2: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 2!")

    # for num_merge = 3
    result = merge(byte_word_count_dict, 3)
    expected = [(b's', b't'), (b'e', b'st'), (b'o', b'w')]
    assert result == expected, f"num_merge = 3: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 3!")

    # for num_merge = 4
    result = merge(byte_word_count_dict, 4)
    expected = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow')]
    assert result == expected, f"num_merge = 4: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 4!")

    # for num_merge = 5
    result = merge(byte_word_count_dict, 5)
    expected = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est')]
    assert result == expected, f"num_merge = 5: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 5!")

    # for num_merge = 6
    result = merge(byte_word_count_dict, 6)
    expected = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    assert result == expected, f"num_merge = 6: Expected {expected}, but got {result}"
    print("Merge test passed for num_merge = 6!")

def test_train_bpe():
    input_path = "/root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.txt"  # Replace with your test file path
    vocab_size = 500
    special_tokens = ["<|endoftext|>", "[MASK]", "\\n"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    print("Vocabulary size:", len(vocab))
    print("Number of merges:", len(merges))
    print("First 10 merges:", merges[:10])


if __name__ == "__main__":
    # test_preprocess_for_bpe()
    # test_merge()
    test_train_bpe()
