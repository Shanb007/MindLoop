import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and feed-forward network.
    
    Args:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads
        ff_mult: Multiplier for feed-forward network dimension
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 12, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Feed-forward network with configurable activation
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.drop_shortcut(attn_out)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = self.drop_shortcut(ff_out)
        x = x + shortcut
        return x


class RecurrentTransformerModel(nn.Module):
    """Recurrent Transformer model for language modeling.
    
    This model processes input through multiple recurrent passes of transformer layers,
    allowing for deeper computation without increasing parameter count proportionally.
    
    Args:
        hidden_dim: Dimension of hidden states
        vocab_size: Size of vocabulary
        num_recurrent_layers: Number of recurrent transformer layers
        num_heads: Number of attention heads
        ff_mult: Multiplier for feed-forward network dimension
        dropout: Dropout probability
        num_recurrences: Number of times to pass through recurrent layers
        seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 50257,
        num_recurrent_layers: int = 8,
        num_heads: int = 12,
        ff_mult: int = 4,
        dropout: float = 0.1,
        num_recurrences: int = 1,
        seq_len: int = 1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.input_transformer = TransformerBlock(
            hidden_dim, num_heads, ff_mult, dropout
        )

        self.recurrent_layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, ff_mult, dropout)
                for _ in range(num_recurrent_layers)
            ]
        )
        self.num_recurrences = num_recurrences

        self.output_transformer = TransformerBlock(
            hidden_dim, num_heads, ff_mult, dropout
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        # Output projection layer
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        # Add positional embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        x = self.input_transformer(x)

        # Apply recurrent transformer layers
        for recurrence_idx in range(self.num_recurrences):
            for layer in self.recurrent_layers:
                x = layer(x)

        x = self.output_transformer(x)

        x = self.final_norm(x)
        logits = self.head(x)

        return logits
    
    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
