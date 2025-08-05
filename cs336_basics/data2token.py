from cs336_basics.tokenizer.BPETokenizer import BPETokenizer
import numpy as np

def data2token(data_path: str, vocab_path: str, merges_path: str, output_path: str):
    """
    Convert the string based data into token IDs using a BPE tokenizer.
    Save the tokenized data to the specified output path by using np.save in order to
    facilitate memory mapping during training.
    """
    import os
    tokenizer = BPETokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)
    
    # Get file size and show info
    file_size = os.path.getsize(data_path)
    print(f"Processing file of size: {file_size:,} bytes")
    
    print("Reading file...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    print(f"Tokenizing {len(text_data):,} characters...")
    token_ids = tokenizer.encode(text_data)
    
    print(f"Converting {len(token_ids):,} tokens to numpy array...")
    token_array = np.array(token_ids, dtype=np.int32)
    np.save(output_path, token_array)
    print(f"Tokenized data saved to {output_path}")

def data2token_streaming(data_path: str, vocab_path: str, merges_path: str, output_path: str, 
                        chunk_size: int = 1024*1024):
    """
    Memory-efficient streaming tokenization for very large files.
    Uses tokenizer.encode_iterable for optimal memory usage.
    """
    import os
    tokenizer = BPETokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)
    
    # Get file size for progress tracking
    file_size = os.path.getsize(data_path)
    print(f"Processing file of size: {file_size:,} bytes")
    
    def text_chunks():
        """Generator that yields text chunks from file with progress tracking"""
        processed_bytes = 0
        with open(data_path, 'r', encoding='utf-8', buffering=chunk_size) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Update progress
                chunk_bytes = len(chunk.encode('utf-8'))
                processed_bytes += chunk_bytes
                progress = min(100.0, (processed_bytes / file_size) * 100)
                
                # Simple progress bar
                bar_length = 50
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: |{bar}| {progress:.1f}% ({processed_bytes:,}/{file_size:,} bytes)', end='', flush=True)
                
                yield chunk
        print()  # New line after progress bar
    
    # Use streaming tokenization
    print("Starting streaming tokenization...")
    token_ids = list(tokenizer.encode_iterable(text_chunks()))
    
    # Save as numpy array
    print(f"Converting {len(token_ids):,} tokens to numpy array...")
    token_array = np.array(token_ids, dtype=np.int32)
    np.save(output_path, token_array)
    print(f"Tokenized data saved: {len(token_array):,} tokens")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert text data to token IDs using BPE tokenizer.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input text data file.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the tokenizer vocabulary file.')
    parser.add_argument('--merges_path', type=str, required=True, help='Path to the tokenizer merges file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the tokenized data.')
    parser.add_argument('--streaming', type=bool, default=True, help='Use streaming tokenization (better for large files).')
    parser.add_argument('--chunk_size', type=int, default=1024*1024, help='Chunk size for streaming (bytes).')

    args = parser.parse_args()
    
    if args.streaming:
        print("Using streaming tokenization...")
        data2token_streaming(args.data_path, args.vocab_path, args.merges_path, args.output_path, args.chunk_size)
    else:
        print("Using standard tokenization...")
        data2token(args.data_path, args.vocab_path, args.merges_path, args.output_path)
    
    print("Data tokenization complete.")