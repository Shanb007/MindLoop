import os
from typing import Tuple, Iterator, Optional
import logging

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for text data with sliding window approach.
    
    Args:
        dataset: HuggingFace dataset object
        tokenizer: Tokenizer for text encoding
        seq_len: Sequence length for each sample
    """
    
    def __init__(self, dataset, tokenizer, seq_len: int = 128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.pad_token_id

    def __iter__(self) -> Iterator[torch.Tensor]:
        for ex in self.dataset["train"]:
            # Encode text with improved handling
            try:
                ids = self.tokenizer.encode(
                    ex["text"],
                    truncation=True,
                    max_length=None,
                    add_special_tokens=True
                )
            except Exception as e:
                logger.warning(f"Error encoding text: {e}")
                continue
            n = len(ids)
            if n < 1:
                continue
            # Sliding window: create all possible chunks of length seq_len
            for i in range(n):
                chunk = ids[max(0, i - self.seq_len + 1) : i + 1]
                # Left-pad to seq_len
                if len(chunk) < self.seq_len:
                    chunk = [self.pad_token_id] * (self.seq_len - len(chunk)) + chunk
                yield torch.tensor(chunk)


def collate_fn(batch: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for language modeling.
    
    Args:
        batch: List of sequences
        
    Returns:
        Tuple of (input, target) tensors
    """
    # Stack sequences into batch tensor
    batch_tensor = torch.stack(batch, dim=0)
    
    # Create input and target sequences (shifted by 1)
    x = batch_tensor[:, :-1]
    y = batch_tensor[:, 1:]
    
    return x, y


def get_dataloader(
    batch_size: int = 4,
    seq_len: int = 128,
    drop_last: bool = True,
    num_workers: int = 0,
    dataset_name: str = "nampdn-ai/tiny-strange-textbooks",
    tokenizer_name: str = "bert-base-uncased",
) -> Tuple[DataLoader, AutoTokenizer]:
    """Create a DataLoader for streaming text data.
    
    Args:
        batch_size: Batch size for training
        seq_len: Sequence length for each sample
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        dataset_name: Name of the HuggingFace dataset
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        Tuple of (DataLoader, tokenizer)
    """
    # Load tokenizer with caching
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir="./cache/tokenizers"
    )
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name,
        token=os.getenv("HF_TOKEN"),
        streaming=True,
        cache_dir="./cache/datasets"
    )
    # Create dataset and dataloader
    ds = StreamingTextDataset(dataset, tokenizer, seq_len=seq_len)
    
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA available
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
    )
    
    logger.info(f"DataLoader created with batch_size={batch_size}, seq_len={seq_len}")
    return dataloader, tokenizer


if __name__ == "__main__":
    # Interactive demo for testing data loading
    dataloader, tokenizer = get_dataloader(batch_size=2, seq_len=128)
    data_iter = iter(dataloader)

    print("\n" + "="*60)
    print("Streaming Dataset Interactive Demo")
    print("Press Enter to load next batch (Ctrl+C to exit)")
    print("="*60 + "\n")
    
    while True:
        input()
        try:
            x, y = next(data_iter)
        except StopIteration:
            print("\n[INFO] End of dataset reached. Restarting iterator...\n")
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        print("\n--- Batch Information ---")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        print("\nEncoded input (first sequence):", x[0][:20].tolist(), "...")
        print("\nDecoded input (first sequence):")
        print(f"  '{tokenizer.decode(x[0], skip_special_tokens=True)[:100]}...'")
        print("\nDecoded target (first sequence):")
        print(f"  '{tokenizer.decode(y[0], skip_special_tokens=True)[:100]}...'")
        print("-" * 60)
