import sys
import os
from typing import Optional, Dict, Any
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_recurrence_evaluation(
    csv_path: str,
    output_dir: str = "evaluation",
    style: str = "darkgrid"
) -> str:
    """
    Create enhanced plots for recurrence evaluation results.

    Args:
        csv_path: Path to the CSV file containing evaluation results
        output_dir: Directory to save the plot
        style: Seaborn plot style
        
    Returns:
        Path to the saved plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Set up the plot style
    sns.set_style(style)
    sns.set_palette("husl")
    plt.figure(figsize=(14, 8))

    # Create the plot with gradient color
    ax = plt.subplot(111)
    cmap = plt.cm.viridis  # type: ignore
    colors = cmap(df["Epoch"] / df["Epoch"].max())

    # Plot main line and scatter points
    ax.plot(
        df["Epoch"], df["Recurrence_Loss"], "-", linewidth=2, alpha=0.7, color="#1f77b4"
    )
    scatter = ax.scatter(
        df["Epoch"],
        df["Recurrence_Loss"],
        c=df["Epoch"],
        cmap=cmap,
        s=80,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add smoothed trendline
    if len(df) > 5:  # Only add trendline if we have enough data points
        window_size = min(10, len(df) // 4)
        rolling_mean = (
            df["Recurrence_Loss"].rolling(window=window_size, center=True).mean()
        )
        ax.plot(
            df["Epoch"],
            rolling_mean,
            "r--",
            linewidth=2.5,
            label=f"Rolling Average (window={window_size})",
        )

    # Calculate percent decrease from start to end
    first_loss = df["Recurrence_Loss"].iloc[0]
    last_loss = df["Recurrence_Loss"].iloc[-1]
    percent_decrease = ((first_loss - last_loss) / first_loss) * 100

    # Set plot title and labels with better styling
    plt.title("Recurrence Loss Over Time", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Epochs", fontsize=14, labelpad=10)
    plt.ylabel("Loss", fontsize=14, labelpad=10)

    # Add annotations
    plt.annotate(
        f"Start: {first_loss:.4f}",
        xy=(df["Epoch"].iloc[0], first_loss),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    plt.annotate(
        f"End: {last_loss:.4f}\n({percent_decrease:.1f}% decrease)",
        xy=(df["Epoch"].iloc[-1], last_loss),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3),
    )

    # Find the minimum loss point
    min_idx = df["Recurrence_Loss"].idxmin()
    min_epoch = df["Epoch"].iloc[min_idx]  # type: ignore
    min_loss = df["Recurrence_Loss"].iloc[min_idx]  # type: ignore

    # Mark the minimum loss point
    plt.annotate(
        f"Min: {min_loss:.4f} (Epoch {min_epoch})",
        xy=(min_epoch, min_loss),
        xytext=(0, -25),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", alpha=0.3),
    )

    # Customize the grid and ticks
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add colorbar for the scatter points
    cbar = plt.colorbar(scatter)
    cbar.set_label("Epoch", fontsize=12)

    # Add legend
    plt.legend(fontsize=12)

    # Show epoch markers at reasonable intervals
    max_ticks = 20
    tick_spacing = max(1, len(df) // max_ticks)
    plt.xticks(df["Epoch"][::tick_spacing], fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "recurrence_loss_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Recurrence loss plot saved to {output_path}")

    return output_path


def plot_training_loss(
    csv_path: str,
    output_dir: str = "evaluation",
    show_statistics: bool = True
) -> str:
    """
    Create an enhanced plot of training loss over epochs.

    Args:
        csv_path: Path to the training loss CSV file
        output_dir: Directory to save the plot
        show_statistics: Whether to show statistics on the plot
        
    Returns:
        Path to the saved plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Set up the plot
    plt.figure(figsize=(12, 7))
    
    # Create main plot
    plt.plot(df["Epoch"], df["Loss"], "o-", linewidth=2.5, markersize=8, 
             label="Training Loss", color="#1f77b4")
    
    # Add rolling average if enough data points
    if len(df) > 5:
        window = min(5, len(df) // 3)
        rolling_avg = df["Loss"].rolling(window=window, center=True).mean()
        plt.plot(df["Epoch"], rolling_avg, "--", linewidth=2, 
                 label=f"Moving Avg (window={window})", color="#ff7f0e", alpha=0.8)

    # Add statistics if requested
    if show_statistics and len(df) > 0:
        final_loss = df["Loss"].iloc[-1]
        min_loss = df["Loss"].min()
        min_epoch = df.loc[df["Loss"].idxmin(), "Epoch"]
        
        # Add text box with statistics
        stats_text = f"Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f} (Epoch {min_epoch})"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set axis limits and labels
    plt.ylim(bottom=0)  # Start loss axis from 0
    plt.title("Training Loss Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "training_loss_plot.png")
    plt.savefig(output_path)
    print(f"Training loss plot saved to {output_path}")

    return output_path


def count_model_params(checkpoint_path: str) -> Dict[str, Any]:
    """Count parameters in a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with parameter counts and model info
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model_state', checkpoint)
    
    total_params = 0
    param_counts = {}
    
    for name, param in model_state.items():
        num_params = param.numel()
        total_params += num_params
        
        # Group parameters by layer type
        layer_type = name.split('.')[0]
        if layer_type not in param_counts:
            param_counts[layer_type] = 0
        param_counts[layer_type] += num_params
    
    # Calculate model size in MB
    model_size_mb = sum(param.numel() * param.element_size() 
                       for param in model_state.values()) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'param_counts_by_layer': param_counts,
        'model_size_mb': model_size_mb,
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'global_step': checkpoint.get('global_step', 'unknown'),
            'best_loss': checkpoint.get('best_loss', 'unknown')
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nMindloop Utility Tools")
        print("=" * 40)
        print("Usage: python random_utils.py <command> [args]")
        print("\nCommands:")
        print("  count_params <checkpoint_path>  - Count model parameters")
        print("  plot_recurrence <csv_path>      - Plot recurrence evaluation results")
        print("  plot_training <csv_path>        - Plot training loss curve")
        print("\nExamples:")
        print("  python random_utils.py count_params checkpoints/checkpoint_best.pt")
        print("  python random_utils.py plot_recurrence evaluation/recurrence_evaluation.csv")
        sys.exit(1)

    command = sys.argv[1]

    if command == "count_params" and len(sys.argv) >= 3:
        checkpoint_path = sys.argv[2]
        try:
            info = count_model_params(checkpoint_path)
            print(f"\nModel Parameter Information:")
            print(f"{'='*40}")
            print(f"Total parameters: {info['total_params']:,}")
            print(f"Model size: {info['model_size_mb']:.2f} MB")
            print(f"\nParameters by layer type:")
            for layer, count in info['param_counts_by_layer'].items():
                print(f"  {layer}: {count:,}")
            print(f"\nCheckpoint info:")
            for key, value in info['checkpoint_info'].items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    elif command == "plot_recurrence" and len(sys.argv) >= 3:
        csv_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "evaluation"
        try:
            output_path = plot_recurrence_evaluation(csv_path, output_dir)
            print(f"Plot saved to: {output_path}")
        except Exception as e:
            print(f"Error creating plot: {e}")
            
    elif command == "plot_training" and len(sys.argv) >= 3:
        csv_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "evaluation"
        try:
            output_path = plot_training_loss(csv_path, output_dir)
            print(f"Plot saved to: {output_path}")
        except Exception as e:
            print(f"Error creating plot: {e}")
            
    else:
        print("\nError: Invalid command or missing arguments")
        print("Run 'python random_utils.py' for usage information")
