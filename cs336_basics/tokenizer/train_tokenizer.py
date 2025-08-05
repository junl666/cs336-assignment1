from cs336_basics.tokenizer.BPETokenizer import train_bpe


def train_tokenizer(input_path: str, vocab_size: int,
                    output_path: str, special_tokens: list[str] = ["<|endoftext|>"]) -> None:
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for token, bytes in vocab.items():
            f.write(f"{token} {bytes}\n")
    with open(output_path.replace(".vocab", ".merges"), "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    print(f"Tokenizer trained and saved to {output_path} and {output_path.replace('.vocab', '.merges')}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Special tokens: {special_tokens}")
    print("Training complete.")
    print("You can now use the tokenizer with the trained vocabulary and merges.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--output_path", type=str, default="tokenizer.vocab", help="Path to save the tokenizer vocabulary.")
    parser.add_argument("--special_tokens", nargs='*', default=["<|endoftext|>"], help="List of special tokens to include in the vocabulary.")

    args = parser.parse_args()
    train_tokenizer(args.input_path, args.vocab_size, args.output_path, args.special_tokens)
    print("Tokenizer training script executed successfully.")
