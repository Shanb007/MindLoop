from datetime import datetime
import itertools
import os
import time
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloader
from model import RecurrentTransformerModel
from tqdm import tqdm, trange

from test import generate_and_print_sample, generate_text_simple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(
    checkpoint: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 32,
    checkpoint_path: str = f"checkpoints/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt",
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
) -> None:
    """Train the RecurrentTransformerModel.
    
    Args:
        checkpoint: Path to checkpoint file to resume from
        epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_path: Path to save checkpoints
        learning_rate: Learning rate for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
    """
    os.makedirs("./checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    loader, tokenizer = get_dataloader(batch_size=batch_size, num_workers=4)
    vocab_size = tokenizer.vocab_size
    # Initialize model with improved configuration
    model = RecurrentTransformerModel(
        vocab_size=vocab_size,
        seq_len=128,
        hidden_dim=768,
        num_heads=12,
        dropout=0.15  # Slightly higher dropout for better generalization
    ).to(device)
    
    logger.info(f"Model initialized with {model.get_num_params():,} parameters")
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_epoch = 0
    # Optionally load checkpoint if provided
    if checkpoint is not None and os.path.isfile(checkpoint):
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        start_epoch = checkpoint_data.get("epoch", 0)
        logger.info(f"Loaded checkpoint from {checkpoint}, starting at epoch {start_epoch+1}")
    model.train()
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * 100, eta_min=1e-5)
    
    global_step = 0
    max_batches_per_epoch = 100
    best_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            itertools.islice(loader, max_batches_per_epoch),
            total=max_batches_per_epoch,
            desc=f"Epoch {epoch+1}/{start_epoch + epochs}"
        )
        
        for batch_i, (x, y) in enumerate(progress_bar):
            optimizer.zero_grad()
            x = x.to(torch.long).to(device)
            y = y.to(torch.long).to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / max_batches_per_epoch
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        # Generate sample text for quality check
        logger.info("Generating sample text...")
        sample_contexts = [
            "Introduction: ",
            "Chapter 1: ",
            "The key to understanding "
        ]
        for context in sample_contexts[:1]:  # Generate one sample per epoch
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context=context,
                seq_len=128,
            )
        # Save checkpoint
        checkpoint_data = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_loss": min(best_loss, avg_epoch_loss),
            "config": {
                "hidden_dim": 768,
                "num_heads": 12,
                "seq_len": 128,
                "learning_rate": learning_rate,
            }
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model separately
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = checkpoint_path.replace('.pt', '_best.pt')
            torch.save(checkpoint_data, best_path)
            logger.info(f"New best model saved to {best_path}")
