import csv
import os
from typing import List, Optional, Tuple, Dict
import json
import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model import RecurrentTransformerModel

# Configure logging
logger = logging.getLogger(__name__)


def generate_text_simple(
    model: RecurrentTransformerModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Generate text using various sampling strategies.
    
    Args:
        model: The language model
        idx: Initial context tokens
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context size
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens with highest probability
        top_p: Keep tokens with cumulative probability < p
        
    Returns:
        Generated token indices
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(
    model: RecurrentTransformerModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    seq_len: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate and print a text sample from the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        start_context: Initial text context
        seq_len: Sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        
    Returns:
        Generated text
    """
    model.eval()
    encoded = tokenizer.encode(
        start_context, add_special_tokens=True, truncation=True, max_length=seq_len
    )
    # Convert to tensor and add batch dimension
    encoded_tensor = torch.tensor([encoded], device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=50,
            context_size=seq_len,
            temperature=temperature,
            top_k=top_k,
        )
    decoded_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(f"Generated: {decoded_text.replace(chr(10), ' ')[:200]}...")
    model.train()
    return decoded_text


def evaluate_recurrence_levels(
    checkpoint: Optional[str] = None,
    recurrence_levels: List[int] = [1, 2, 4, 6, 8, 12, 24],
    num_samples: int = 500,
    batch_size: int = 32,
    output_dir: str = "./evaluation",
    save_detailed_results: bool = True,
) -> List[Tuple[int, float]]:
    """
    Evaluates model performance at different recurrence levels.

    Args:
        checkpoint: Path to the model checkpoint
        recurrence_levels: List of recurrence levels to evaluate
        num_samples: Number of examples to evaluate on
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        save_detailed_results: Whether to save detailed evaluation results
        
    Returns:
        List of (recurrence_level, loss) tuples
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    loader, tokenizer = get_dataloader(batch_size=batch_size, num_workers=4)
    vocab_size = tokenizer.vocab_size

    # Validate checkpoint
    if checkpoint is None or not os.path.isfile(checkpoint):
        logger.error("No valid checkpoint provided. Please provide a valid checkpoint.")
        return []

    # Prepare output files
    csv_path = os.path.join(output_dir, "recurrence_evaluation.csv")
    json_path = os.path.join(output_dir, "recurrence_evaluation.json")
    
    results = []
    detailed_results = {}

    # Write CSV header
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Recurrence_Level", "Loss", "Perplexity"])

        # Evaluate each recurrence level
        for recurrence in recurrence_levels:
            logger.info(f"Evaluating model with recurrence level {recurrence}")

            # Create model with current recurrence level
            model = RecurrentTransformerModel(
                vocab_size=vocab_size,
                num_recurrences=recurrence,
                seq_len=128,
                hidden_dim=768,
                num_heads=12,
            ).to(device)

            # Load checkpoint weights
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data["model_state"], strict=False)
            logger.info(f"Loaded checkpoint weights for recurrence level {recurrence}")

            # Set model to eval mode
            model.eval()

            # Set recurrence level
            model.num_recurrences = recurrence

            # Calculate loss on samples
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            samples_processed = 0

            with torch.no_grad():
                data_iter = iter(loader)
                pbar = tqdm(total=num_samples, desc=f"Recurrence {recurrence}")

                while samples_processed < num_samples:
                    try:
                        inputs, targets = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        inputs, targets = next(data_iter)

                    inputs = inputs.to(torch.long).to(device)
                    targets = targets.to(torch.long).to(device)

                    # Forward pass
                    logits = model(inputs)  # [batch, seq_len, vocab]

                    # Reshape for loss calculation
                    batch_size, seq_len = targets.size()
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)

                    # Calculate loss
                    loss = criterion(logits_flat, targets_flat)

                    # Update statistics
                    batch_samples = inputs.size(0)
                    samples_to_add = min(batch_samples, num_samples - samples_processed)
                    total_loss += loss.item() * samples_to_add
                    samples_processed += samples_to_add
                    pbar.update(samples_to_add)

                    if samples_processed >= num_samples:
                        break

            # Calculate metrics
            avg_loss = total_loss / num_samples
            perplexity = np.exp(avg_loss)
            
            logger.info(
                f"Recurrence level {recurrence} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}"
            )

            # Store results
            csv_writer.writerow([recurrence, avg_loss, perplexity])
            results.append((recurrence, avg_loss))
            
            if save_detailed_results:
                detailed_results[str(recurrence)] = {
                    "loss": avg_loss,
                    "perplexity": perplexity,
                    "num_samples": num_samples,
                    "model_params": model.get_num_params(),
                }

    logger.info(f"Evaluation results saved to {csv_path}")
    
    # Save detailed results as JSON
    if save_detailed_results:
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {json_path}")

    # Create enhanced plot
    if results:
        recurrences, losses = zip(*results)
        perplexities = [np.exp(loss) for loss in losses]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        ax1.plot(recurrences, losses, marker="o", linestyle="-", linewidth=2, markersize=8)
        ax1.set_title("Model Loss vs Recurrence Level", fontsize=14)
        ax1.set_xlabel("Recurrence Level", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(recurrences)
        
        # Perplexity plot
        ax2.plot(recurrences, perplexities, marker="s", linestyle="-", linewidth=2, markersize=8, color="orange")
        ax2.set_title("Model Perplexity vs Recurrence Level", fontsize=14)
        ax2.set_xlabel("Recurrence Level", fontsize=12)
        ax2.set_ylabel("Perplexity", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(recurrences)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "recurrence_evaluation_plots.png")
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Plots saved to {plot_path}")
        plt.close()

    return results
