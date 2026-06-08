"""
SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
Treats user history as a sequence and uses causal self-attention to predict next item.

Key difference from SVD/CF: captures order and recency of interactions,
not just co-occurrence. Critical for session-based recommendation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SASRec(nn.Module):
    def __init__(self, n_items: int, hidden_dim: int = 64, n_heads: int = 2,
                 n_layers: int = 2, max_seq_len: int = 50, dropout: float = 0.2):
        """
        n_items: vocabulary size (number of movies)
        hidden_dim: embedding dimension
        n_heads: attention heads
        n_layers: transformer blocks
        max_seq_len: maximum history length to consider
        """
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            SASRecBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item ids, returns (B, L, D) representations"""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Returns scores for candidate items given sequence context.
        seq: (B, L) item ids
        candidates: (B, C) or (C,) candidate item ids
        Returns: (B, C) scores
        """
        h = self.forward(seq)
        # use the last non-padding position
        h_last = h[:, -1, :]  # (B, D)
        if candidates.dim() == 1:
            cand_emb = self.item_emb(candidates)  # (C, D)
            scores = torch.matmul(h_last, cand_emb.T)  # (B, C)
        else:
            cand_emb = self.item_emb(candidates)  # (B, C, D)
            scores = torch.bmm(cand_emb, h_last.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores


class SASRecBlock(nn.Module):
    """Single transformer block with causal masking"""
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention + FFN with pre-norm"""
        L = x.size(1)
        # causal mask: upper triangle is -inf so position i can only attend to j <= i
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        # pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = residual + attn_out
        # pre-norm FFN
        x = x + self.ff(self.norm2(x))
        return x
