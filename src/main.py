import argparse
import datetime
import os
import glob
import sys
from typing import Optional
import logging

from train import train
from test import evaluate_recurrence_levels
from random_utils import plot_recurrence_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_latest_checkpoint() -> Optional[str]:
    """Find the most recent checkpoint file in the checkpoints directory.
    
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = "./checkpoints"
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def main():
    """Main entry point for the mindloop CLI."""
    parser = argparse.ArgumentParser(
        description="Mindloop - Recurrent Transformer Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Train model for 5 epochs
  %(prog)s train -e 5 -b 32
  
  # Evaluate model at different recurrence levels
  %(prog)s evaluate -r 1 2 4 8 16
  
  # Plot existing evaluation results
  %(prog)s plot evaluation/recurrence_evaluation.csv
"""
    )
    # Add version argument
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 0.2.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Available commands"
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train the recurrent transformer model",
        description="Train the model with specified hyperparameters"
    )
    train_parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=f"./checkpoints/checkpoint-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
        help="Path to save the checkpoint file (default: ./checkpoints/checkpoint-<timestamp>.pt)",
    )
    train_parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    train_parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    train_parser.add_argument(
        "-r", "--resume",
        type=str,
        help="Resume training from checkpoint"
    )
    train_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model at different recurrence levels",
        description="Test model performance with varying recurrence depths"
    )
    evaluate_parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=get_latest_checkpoint(),
        help="Path to the checkpoint file (default: latest checkpoint in ./checkpoints)",
    )
    evaluate_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500)",
    )
    evaluate_parser.add_argument(
        "-r",
        "--recurrence-levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 24],
        help="Recurrence levels to evaluate (default: 1 2 4 6 8 12 24)",
    )
    evaluate_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./evaluation",
        help="Directory to save evaluation results (default: ./evaluation)",
    )
    
    # Plot subcommand
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots from evaluation results",
        description="Create visualization plots from CSV evaluation data"
    )
    plot_parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file with evaluation results"
    )
    plot_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./evaluation",
        help="Directory to save plots (default: ./evaluation)"
    )

    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./evaluation", exist_ok=True)
    os.makedirs("./cache", exist_ok=True)

    try:
        if args.mode == "train":
            # Handle resume functionality
            checkpoint_to_load = args.resume if hasattr(args, 'resume') and args.resume else None
            
            train(
                checkpoint=checkpoint_to_load,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_path=args.checkpoint,
                learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 1e-3,
            )
        elif args.mode == "evaluate":
            if args.checkpoint is None:
                logger.error("No checkpoint found. Please train a model first or specify a checkpoint.")
                sys.exit(1)
                
            evaluate_recurrence_levels(
                checkpoint=args.checkpoint,
                recurrence_levels=args.recurrence_levels,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
            
        elif args.mode == "plot":
            plot_recurrence_evaluation(
                csv_path=args.csv_path,
                output_dir=args.output_dir
            )
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
