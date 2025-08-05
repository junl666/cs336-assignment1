#!/usr/bin/env python3
"""
Training script for language model.
Integrates all components: model, optimizer, data loading, checkpointing, etc.
"""

import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

# Import our components
from cs336_basics.dataloader import data_loading, save_checkpoint, load_checkpoint
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.optimizer import AdamW
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_scheduler import learning_rate_schedule
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenizer.BPETokenizer import BPETokenizer


def load_data(data_path: str) -> np.memmap[int]:
    """Load data using memory mapping for efficiency."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Use memory mapping for large datasets
    data = np.memmap(data_path, dtype=np.int32, mode='r')
    return data


def evaluate_model(model, data, batch_size: int, context_length: int, 
                  device: str, num_eval_batches: int = 100):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            inputs, targets = data_loading(data, batch_size, context_length, device)
            logits = model(inputs)
            
            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = cross_entropy_loss(logits, targets)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_eval_batches


def train():
    print("---- Training Language Model ----")
    parser = argparse.ArgumentParser(description='Train a language model')
    
    # Debug mode
    parser.add_argument('--debug_ratio', type=float, default=1.0, help='Ratio of data to use in debug mode (0 < ratio <= 1)')

    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE scaling factor')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feedforward layer dimension')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate (deprecated, use max_lr)')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--decay_steps', type=int, default=50000, help='Number of decay steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations')
    
    # Data and checkpointing
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to tokenizer vocabulary file')
    parser.add_argument('--merges_path', type=str, required=True, help='Path to tokenizer merges file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, help='Path to validation data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--load_checkpoint', type=str, help='Path to checkpoint to load')
    
    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=100, help='Log every N iterations')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluate every N iterations')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint every N iterations')
    parser.add_argument('--num_eval_batches', type=int, default=100, help='Number of batches for evaluation')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--disable_tensorboard', action='store_true', help='Disable TensorBoard logging')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = None
    if not args.disable_tensorboard:
        os.makedirs(args.log_dir, exist_ok=True)
        # Create unique log directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir {args.log_dir}")
    
    # Load tokenizer and vocabulary
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path
    )
    print("Tokenizer loaded.")

    # Load data
    print("Loading training data...")
    train_data = load_data(args.train_data)

    # If in debug mode, use only a subset of the data
    if args.debug_ratio < 1.0:
        print(f"Debug mode: using {args.debug_ratio*100:.1f}% of the training data")
        total_tokens = len(train_data)
        subset_size = int(total_tokens * args.debug_ratio)
        train_data = train_data[:subset_size] # TODO: handle random sampling if needed
    
    val_data = None
    if args.val_data:
        print("Loading validation data...")
        val_data = load_data(args.val_data)
    
    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        device=device
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,  # Start with max_lr, will be adjusted by scheduler
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if specified
    start_iter = 0
    max_iters = args.max_iters
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        start_iter = load_checkpoint(args.load_checkpoint, model, optimizer)
        max_iters += start_iter
        print(f"Resuming from iteration {start_iter}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    print(f"Learning rate schedule: max_lr={args.max_lr}, min_lr={args.min_lr}, warmup_steps={args.warmup_steps}, decay_steps={args.decay_steps}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for iteration in range(start_iter, max_iters):
        start_time = time.time()
        
        # Update learning rate using scheduler
        current_lr = learning_rate_schedule(
            t=iteration,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps
        )
        
        # Apply learning rate to optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Get batch
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        # print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        logits = model(inputs) # shape: (batch_size, context_length, vocab_size)
        
        # Reshape for loss computation
        logits = rearrange(logits, 'b s v -> (b s) v')  # shape: (batch_size * context_length, vocab_size)
        targets = rearrange(targets, 'b s -> (b s)')  # shape: (batch_size * context_length,)
        
        # Compute loss
        loss = cross_entropy_loss(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(list(model.parameters()), args.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration:6d} | Loss: {loss.item():.4f} | "
                  f"Time: {elapsed:.3f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/Train', loss.item(), iteration)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar('Time_per_Batch', elapsed, iteration)
        
        # Evaluation
        if val_data is not None and iteration % args.eval_interval == 0:
            print("Evaluating...")
            eval_loss = evaluate_model(
                model, val_data, args.batch_size, args.context_length, 
                device, args.num_eval_batches
            )
            print(f"Iter {iteration:6d} | Val Loss: {eval_loss:.4f}")
            
            # Log validation loss to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/Validation', eval_loss, iteration)
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    print(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("TensorBoard writer closed.")
    
    print("Training completed!")


if __name__ == "__main__":
    train()