#!/usr/bin/env python3
"""
Text generation and decoding utilities for language models.
Supports temperature scaling, top-p sampling, and checkpoint loading.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
import argparse
import os

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenizer.BPETokenizer import BPETokenizer
from cs336_basics.dataloader import load_checkpoint


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits and return softmax probabilities.
    
    Args:
        logits: Raw model outputs of shape (..., vocab_size)
        temperature: Temperature parameter for scaling. Lower values make distribution more peaked.
                    τ → 0 makes it more deterministic, τ → ∞ makes it more uniform.
    
    Returns:
        Probability distribution after temperature scaling and softmax
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    probabilities = F.softmax(scaled_logits, dim=-1)
    
    return probabilities


def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to probability distribution.
    
    Args:
        probabilities: Probability distribution of shape (..., vocab_size)
        p: Nucleus probability threshold (0 < p <= 1)
    
    Returns:
        Modified probability distribution with low-probability tokens set to 0
    """
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the cutoff index where cumulative probability exceeds p
    # We keep indices where cumulative probability <= p, plus the first one that exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    
    # Keep at least the first token (the most probable one)
    sorted_indices_to_remove[..., 0] = False
    
    # Create a mask for the original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # Set probabilities of removed indices to 0
    filtered_probabilities = probabilities.clone()
    filtered_probabilities[indices_to_remove] = 0.0
    
    # Renormalize
    filtered_probabilities = filtered_probabilities / filtered_probabilities.sum(dim=-1, keepdim=True)
    
    return filtered_probabilities


def sample_next_token(logits: torch.Tensor, 
                     temperature: float = 1.0, 
                     top_p: Optional[float] = None) -> torch.Tensor:
    """
    Sample next token from model logits using temperature scaling and optional top-p sampling.
    
    Args:
        logits: Model output logits of shape (..., vocab_size)
        temperature: Temperature for scaling
        top_p: Top-p threshold for nucleus sampling (None to disable)
    
    Returns:
        Sampled token indices
    """
    # Apply temperature scaling
    probabilities = softmax_with_temperature(logits, temperature)
    
    # Apply top-p sampling if specified
    if top_p is not None:
        probabilities = top_p_sampling(probabilities, top_p)
    
    # Sample from the distribution
    next_token = torch.multinomial(probabilities, num_samples=1)
    
    return next_token.squeeze(-1)


def load_model_from_checkpoint(checkpoint_path: str, 
                             vocab_size: int,
                             d_model: int,
                             n_layers: int,
                             n_heads: int,
                             d_ff: int,
                             context_length: int,
                             max_seq_len: int = 1024,
                             rope_theta: float = 10000.0,
                             device: str = 'auto') -> TransformerLM:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        context_length: Context length
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        device: Device to load model on
    
    Returns:
        Loaded TransformerLM model
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully. Trained for {checkpoint.get('iteration', 'unknown')} iterations.")
    
    return model


def generate_text(model: TransformerLM,
                 tokenizer: BPETokenizer,
                 prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_p: Optional[float] = None,
                 device: str = 'auto') -> str:
    """
    Generate text from a trained language model.
    
    Args:
        model: Trained TransformerLM model
        tokenizer: BPE tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling (lower = more deterministic)
        top_p: Top-p threshold for nucleus sampling (None to disable)
        device: Device to run generation on
    
    Returns:
        Generated text string
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Convert to tensor
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Get special token IDs
    eos_token_id = None
    if '<|endoftext|>' in tokenizer.special_tokens:
        # Encode the special token to get its ID
        eos_tokens = tokenizer.encode('<|endoftext|>')
        if eos_tokens:
            eos_token_id = eos_tokens[0]  # Usually special tokens are single tokens
    
    print(f"Generating text with prompt: '{prompt}'")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Max new tokens: {max_new_tokens}")
    print("-" * 50)
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get model predictions
            # Only use the last context_length tokens to avoid exceeding model capacity
            if input_ids.size(1) > model.context_length:
                model_input = input_ids[:, -model.context_length:]
            else:
                model_input = input_ids
            
            # Forward pass
            logits = model(model_input)  # Shape: (batch_size, seq_len, vocab_size)
            
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            
            # Sample next token
            next_token = sample_next_token(next_token_logits, temperature, top_p)
            
            # Check for end of sequence
            if eos_token_id is not None and next_token.item() == eos_token_id:
                print(f"Generated <|endoftext|> token, stopping generation.")
                break
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Append to input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Print progress every 10 tokens
            if (step + 1) % 10 == 0:
                partial_text = tokenizer.decode(generated_tokens)
                print(f"Step {step + 1:3d}: ...{partial_text[-50:]}")
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    return prompt + generated_text


def interactive_generation(model: TransformerLM, 
                         tokenizer: BPETokenizer,
                         device: str = 'auto'):
    """
    Interactive text generation interface.
    
    Args:
        model: Trained model
        tokenizer: BPE tokenizer  
        device: Device to run on
    """
    print("=== Interactive Text Generation ===")
    print("Type 'quit' to exit, 'help' for commands")
    print()
    
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("Commands:")
                print("  quit - Exit the program")
                print("  help - Show this help")
                print("Parameters can be set by typing: temp=0.8 top_p=0.9 max_tokens=50")
                continue
            elif not prompt:
                continue
            
            # Parse parameters from input
            temperature = 1.0
            top_p = 0.9
            max_tokens = 100
            
            # Simple parameter parsing
            parts = prompt.split()
            actual_prompt_parts = []
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key == 'temp' or key == 'temperature':
                        temperature = float(value)
                    elif key == 'top_p':
                        top_p = float(value) if value.lower() != 'none' else None
                    elif key == 'max_tokens':
                        max_tokens = int(value)
                else:
                    actual_prompt_parts.append(part)
            
            actual_prompt = ' '.join(actual_prompt_parts)
            
            if not actual_prompt:
                print("Please provide a prompt!")
                continue
            
            # Generate text
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=actual_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device
            )
            
            print(f"\nGenerated text:\n{result}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate text from trained language model')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to tokenizer vocabulary')
    parser.add_argument('--merges_path', type=str, required=True,
                       help='Path to tokenizer merges')
    
    # Model architecture (should match training configuration)
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, 
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (lower = more deterministic)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p threshold for nucleus sampling (set to 1.0 to disable)')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, etc.)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_path}")
    
    if not os.path.exists(args.merges_path):
        raise FileNotFoundError(f"Merges file not found: {args.merges_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path
    )
    print("Tokenizer loaded.")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        device=args.device
    )
    
    # Run generation
    if args.interactive:
        interactive_generation(model, tokenizer, args.device)
    else:
        if not args.prompt:
            raise ValueError("Must provide --prompt for non-interactive generation")
        
        # Handle top_p parameter
        top_p = args.top_p if args.top_p < 1.0 else None
        
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=top_p,
            device=args.device
        )
        
        print("\n" + "="*50)
        print("GENERATED TEXT:")
        print("="*50)
        print(result)
        print("="*50)


if __name__ == "__main__":
    main()
