import os
from typing import BinaryIO, Union, Iterable, Iterator
import regex as re
import multiprocessing as mp
from functools import partial
import json


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
    Return the count of each byte word in the text.
    TODO: Make it suitable for BPETokenizer.
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
        num_processes = 10  # Use all available CPU cores
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


class BPETokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, 
        list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer 
        from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f]
        
        return BPETokenizer(vocab, merges, special_tokens)

    def _split_by_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        """
        Split text on special tokens while preserving the special tokens.
        Handles overlapping special tokens by prioritizing longer tokens.
        
        Args:
            text: Input text
            special_tokens: List of special tokens to split on
            
        Returns:
            List of text parts, with special tokens preserved as separate elements
            
        Example:
            text = "Hello world! <|endoftext|> Great!" 
            special_tokens = ["<|endoftext|>"]
            result = ['Hello world! ', '<|endoftext|>', ' Great!']
        """
        if not special_tokens:
            return [text]
        
        # 按长度降序排序，确保较长的token优先匹配
        special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
        
        # 使用贪婪匹配策略：从左到右扫描，优先匹配最长的特殊标记
        result = []
        i = 0
        
        while i < len(text):
            found_token = None
            found_length = 0
            
            # 检查从当前位置开始是否匹配任何特殊标记
            for token in special_tokens_sorted:
                if text[i:].startswith(token):
                    found_token = token
                    found_length = len(token)
                    break  # 由于按长度排序，第一个匹配的就是最长的
            
            if found_token:
                # 如果前面有非特殊标记的文本，先添加到结果
                if i > 0 and (not result or result[-1] != ''):
                    # 找到上一个特殊标记的结束位置
                    last_pos = 0
                    for j in range(len(result)):
                        if result[j] in special_tokens:
                            # 计算这个特殊标记在原文本中的位置
                            current_pos = 0
                            for k in range(j + 1):
                                current_pos += len(result[k])
                            last_pos = current_pos
                    
                    if i > last_pos:
                        prefix = text[last_pos:i]
                        if prefix:
                            result.append(prefix)
                
                # 添加找到的特殊标记
                result.append(found_token)
                i += found_length
            else:
                # 没有匹配到特殊标记，跳到下一个字符
                i += 1
        
        # 处理剩余的文本
        if result:
            # 计算所有已处理文本的长度
            processed_length = 0
            for part in result:
                processed_length += len(part)
            
            if processed_length < len(text):
                remaining = text[processed_length:]
                if remaining:
                    result.append(remaining)
        else:
            # 没有找到任何特殊标记，返回原文本
            result = [text]
        
        # 过滤掉空字符串，但保持特殊标记
        filtered_result = []
        for part in result:
            if part or part in special_tokens:
                filtered_result.append(part)
        
        return filtered_result

    def _pretokenize(self, text: str, special_tokens: list[str]) -> Iterator[Union[tuple[bytes], bytes]]:
        """
        Pre-tokenize the input text into byte words.
        This is a helper function that converts the text into a list of byte words.
        This function should not drop any special tokens.
        """
        # 使用改进的分割函数来处理重叠的特殊标记
        parts = self._split_by_special_tokens(text, special_tokens)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for part in parts:
            if not part:  # 跳过空字符串
                continue
                
            if part in special_tokens:
                # 特殊标记作为整体保留
                byte_word = part.encode('utf-8')
                yield byte_word
            else:
                # 普通文本使用PAT进行tokenization
                tokens = re.findall(PAT, part)
                for token_text in tokens:
                    if token_text:
                        # Convert each token to bytes
                        byte_word = tuple(bytes([b]) for b in token_text.encode('utf-8'))
                        yield byte_word

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        Step 1: Pre-tokenize the text into byte words.
        Step 2: Apply BPE merges to the byte words and return the token IDs.
        This function should yield token IDs for each byte word.
        """
        assert self.vocab is not None, "Vocabulary must be initialized before encoding"
        assert self.merges is not None, "Merges must be initialized before encoding"
        # Step 1: Pre-tokenize the text into byte words
        byte_words = self._pretokenize(text, self.special_tokens)
        bytes_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        # Step 2: Apply BPE merges to the byte words
        encoded_ids = []
        for byte_word in byte_words:
            if isinstance(byte_word, bytes):
                # This is a special token
                if byte_word in bytes_special_tokens:
                    # Convert special token to its ID
                    token_id = self.reversed_vocab.get(byte_word, None)
                    if token_id is not None:
                        encoded_ids.append(token_id)
            elif isinstance(byte_word, tuple):
                # This is a byte word (tuple of bytes)
                for token_id in self._merge_byte_word(byte_word):
                    encoded_ids.append(token_id)
            else:
                raise ValueError(f"Unexpected byte word type: {type(byte_word)}")
        return encoded_ids
            
    def _merge_byte_word(self, byte_word: tuple[bytes, ...]) -> Iterator[int]:
        """
        Merge a byte word using the BPE merges defined in self.merges.
        This function should yield the token IDs for the merged byte word.
        Uses merge_priority for efficient merging.
        """
        # Convert tuple of bytes to a list for easier manipulation
        word = list(byte_word)
        
        # Keep applying merges until no more merges are possible
        while len(word) >= 2:
            # Find all possible merge pairs in the current word
            possible_merges = []
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_priority:
                    # Store the merge pair, its position, and its priority
                    possible_merges.append((self.merge_priority[pair], i, pair))
            
            if not possible_merges:
                # No more merges possible
                break
            
            # Sort by priority (lower number = higher priority, applied earlier)
            possible_merges.sort(key=lambda x: x[0])
            
            # Apply the highest priority merge
            _, merge_pos, merge_pair = possible_merges[0]
            
            # Merge the two bytes at the specified position
            merged_bytes = merge_pair[0] + merge_pair[1]
            word[merge_pos] = merged_bytes
            word.pop(merge_pos + 1)
            
            # Continue with the updated word
        
        # Convert the final merged word to token IDs
        for token_bytes in word:
            # Look up the token ID in the vocabulary
            token_id = self.reversed_vocab.get(token_bytes, None)
            
            if token_id is not None:
                yield token_id
            else:
                # If not found in vocab, break down to individual bytes
                # This handles unknown tokens by falling back to byte-level
                if isinstance(token_bytes, bytes):
                    for byte_val in token_bytes:
                        # Single bytes should be in vocab with their byte value as ID
                        single_byte = bytes([byte_val])
                        single_byte_id = self.reversed_vocab.get(single_byte, byte_val)
                        yield single_byte_id
                else:
                    # This shouldn't happen, but handle it gracefully
                    raise ValueError(f"Unexpected token type: {type(token_bytes)}")
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is 
        required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text. Note that 
        input IDs are not guaranteed to map to valid Unicode strings.
        """
        # copied from ...
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]
            else:
                token = bytes(replacement_char, encoding='utf-8')

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 


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
    input_path = "/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt"  # Replace with your test file path
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    print("Vocabulary size:", len(vocab))
    print("Number of merges:", len(merges))
    print("First 10 merges:", merges[:10])

    # TODO: Serialize the resulting vocabulary and merges to disk for further inspection

def test_bpetokenizer_pretokenize():
    text1 = "Hello world! <|endoftext|>Great!"
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer({}, [])
    pretokenized1 = list(tokenizer._pretokenize(text1, special_tokens))

    expected1 = [
        (b'H', b'e', b'l', b'l', b'o'),
        (b' ', b'w', b'o', b'r', b'l', b'd'),
        (b'!',),
        (b' ',),
        b'<|endoftext|>',
        (b'G', b'r', b'e', b'a', b't'),
        (b'!',)
    ]

    assert pretokenized1 == expected1, f"Expected {expected1}, but got {pretokenized1}"

    text2 = "Hello Hello world!"
    pretokenized2 = list(tokenizer._pretokenize(text2, special_tokens))
    expected2 = [
        (b'H', b'e', b'l', b'l', b'o'),
        (b' ', b'H', b'e', b'l', b'l', b'o'),
        (b' ', b'w', b'o', b'r', b'l', b'd'),
        (b'!',)
    ]

    assert pretokenized2 == expected2, f"Expected {expected2}, but got {pretokenized2}"

    print("Pre-tokenization test passed!")

def test_overlapping_special_tokens():
    """
    Test handling of overlapping special tokens like <|endoftext|> and <|endoftext|><|endoftext|>
    """
    # 创建一个简单的tokenizer来测试
    vocab = {i: bytes([i]) for i in range(256)}  # 基础字节词汇表
    vocab[256] = b'<|endoftext|>'
    vocab[257] = b'<|endoftext|><|endoftext|>'
    
    merges = []
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    
    # 测试文本
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    
    # 首先测试预标记化
    pretokenized = list(tokenizer._pretokenize(test_string, special_tokens))
    print("Pretokenized:", pretokenized)
    
    # 验证双重特殊标记被正确识别
    assert b'<|endoftext|><|endoftext|>' in pretokenized
    assert pretokenized.count(b'<|endoftext|><|endoftext|>') == 1
    assert pretokenized.count(b'<|endoftext|>') == 1
    
    # 测试编码
    ids = tokenizer.encode(test_string)
    print("Encoded IDs:", ids)
    
    # 测试解码
    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)
    
    # 验证往返一致性
    assert decoded == test_string
    
    print("Overlapping special tokens test passed!")

def test_bpetokenizer_encode():
    # 示例使用
    vocab = {
        0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 
        5: b't', 6: b'th', 7: b' c', 8: b' a', 
        9: b'the', 10: b' at'
    }

    merges = [
        (b't', b'h'), 
        (b' ', b'c'), 
        (b' ', b'a'), 
        (b'th', b'e'), 
        (b' a', b't')
    ]

    tokenizer = BPETokenizer(vocab, merges)

    # 测试示例
    text = "the cat ate"
    encoded = tokenizer.encode(text)
    assert list(encoded) == [9, 7, 1, 5, 10, 3], "Encoding did not match expected output"
    print("Encoding test passed!")

if __name__ == "__main__":
    # test_preprocess_for_bpe()
    # test_merge()
    # test_train_bpe()
    # test_bpetokenizer_pretokenize()
    test_bpetokenizer_encode()
    test_overlapping_special_tokens()
    print("All tests passed!")
