# mindloop

**Recurrent Transformer Language Model for Deep Language Understanding**

A group project for CS5804 exploring the intersection of recurrent processing and transformer architectures for improved language modeling capabilities.

## 🎯 Project Overview

Mindloop implements a novel **Recurrent Transformer Model** that processes input through multiple recurrent passes of transformer layers, allowing for deeper computation without proportionally increasing parameter count. This architecture explores how iterative refinement can enhance language understanding and generation.

### Key Features

- **🔄 Recurrent Transformer Architecture**: Multiple passes through the same transformer layers for iterative refinement
- **🎛️ Advanced Training Pipeline**: Learning rate scheduling, gradient clipping, and label smoothing
- **📊 Comprehensive Evaluation**: Multi-level recurrence analysis with loss and perplexity metrics
- **🎲 Sophisticated Text Generation**: Temperature, top-k, and top-p sampling strategies
- **📈 Rich Visualization**: Interactive plots for training progress and evaluation results
- **⚡ Production-Ready**: Professional logging, error handling, and checkpoint management

## 🏗️ Architecture

The model consists of three main components:

1. **Input Transformer Block**: Initial processing of embedded sequences
2. **Recurrent Transformer Layers**: Core layers that process input multiple times
3. **Output Transformer Block**: Final processing before language modeling head

The recurrence mechanism allows the model to iteratively refine its representations, potentially capturing more complex linguistic patterns.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional but recommended)
- HuggingFace account token (for dataset access)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mindloop

# Install dependencies
uv sync --extra cu124  # For CUDA 12.4
# or
uv sync --extra cpu     # For CPU-only

# Set up environment variables
cp .env.example .env
# Edit .env and add your HuggingFace token
```

### Basic Usage

```bash
# View all available commands
uv run src/main.py --help

# Train the model
uv run src/main.py train -e 10 -b 32

# Evaluate at different recurrence levels
uv run src/main.py evaluate -r 1 2 4 8 16

# Generate visualization plots
uv run src/main.py plot evaluation/recurrence_evaluation.csv
```

## 📋 Detailed Commands

### Training

```bash
# Basic training
uv run src/main.py train -e 5 -b 32

# Training with custom learning rate
uv run src/main.py train -e 10 -lr 5e-4

# Resume from checkpoint
uv run src/main.py train -r checkpoints/checkpoint_best.pt -e 5

# Background training with logging
nohup uv run src/main.py train -e 50 > training.log 2>&1 &
```

### Evaluation

```bash
# Evaluate with default recurrence levels [1, 2, 4, 6, 8, 12, 24]
uv run src/main.py evaluate

# Custom recurrence levels
uv run src/main.py evaluate -r 1 2 4 8 16 32

# Specify number of evaluation samples
uv run src/main.py evaluate -n 1000 -b 16
```

### Utilities

```bash
# Count model parameters
uv run src/random_utils.py count_params checkpoints/checkpoint_best.pt

# Generate training loss plot
uv run src/random_utils.py plot_training evaluation/training_loss.csv

# Generate recurrence evaluation plot
uv run src/random_utils.py plot_recurrence evaluation/recurrence_evaluation.csv
```

## 📁 Project Structure

```
mindloop/
├── src/
│   ├── main.py           # CLI interface and entry point
│   ├── model.py          # Recurrent transformer architecture
│   ├── train.py          # Training pipeline with advanced features
│   ├── test.py           # Evaluation and text generation
│   ├── data.py           # Streaming dataset and data loading
│   └── random_utils.py   # Visualization and utility functions
├── checkpoints/          # Model checkpoints (auto-created)
├── evaluation/           # Evaluation results and plots (auto-created)
├── cache/               # Dataset and tokenizer cache (auto-created)
├── report/              # Project documentation
├── pyproject.toml       # Project configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🧪 Experimental Features

### Recurrence Levels
The model supports configurable recurrence levels (1-32+), allowing exploration of the trade-off between computational depth and performance:

- **Level 1**: Standard transformer (baseline)
- **Levels 2-8**: Sweet spot for most tasks
- **Levels 12+**: Deep recurrence for complex patterns

### Sampling Strategies
Advanced text generation with multiple sampling approaches:

- **Greedy**: Deterministic, highest probability tokens
- **Temperature**: Controlled randomness (0.1 = conservative, 2.0 = creative)
- **Top-k**: Sample from top k most probable tokens
- **Top-p (Nucleus)**: Sample from tokens with cumulative probability < p

## 📊 Model Performance

The model is evaluated on the "tiny-strange-textbooks" dataset with metrics including:

- **Training Loss**: Cross-entropy with label smoothing
- **Perplexity**: Exponential of cross-entropy loss
- **Recurrence Analysis**: Performance across different recurrence depths

Sample results show that moderate recurrence levels (4-8) often provide the best performance-efficiency trade-off.

## 🔧 Configuration

### Model Hyperparameters
- Hidden dimension: 768
- Attention heads: 12
- Sequence length: 128
- Dropout: 0.15
- Feed-forward multiplier: 4

### Training Configuration
- Optimizer: AdamW with weight decay
- Learning rate: 1e-3 with cosine annealing
- Batch size: 32 (configurable)
- Gradient clipping: 1.0
- Label smoothing: 0.1

## 📈 Monitoring and Analysis

### Training Monitoring
- Real-time loss tracking with progress bars
- Sample text generation during training
- Best model checkpoint saving
- Comprehensive logging

### Evaluation Analysis
- CSV export of all metrics
- JSON detailed results
- Interactive visualization plots
- Statistical analysis tools

## 🤝 Contributing

This project was developed as part of CS5804. The architecture explores novel applications of recurrent processing in transformer models.

### Development Setup

```bash
# Install development dependencies
uv sync --extra cu124 --dev

# Run type checking
mypy src/

# Format code
black src/

# Run tests
pytest tests/
```

## 🔬 Research Context

This implementation explores several research questions:

1. **Depth vs. Parameters**: Can recurrent processing achieve deeper computation without proportional parameter growth?
2. **Iterative Refinement**: How does multiple processing of the same input affect representation quality?
3. **Computational Efficiency**: What is the optimal recurrence level for different tasks?

## 📝 License

This project is developed for educational purposes as part of CS5804.


---

**Version**: 0.2.0  
**Last Updated**: 2025-07-16