import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttentionPooling(nn.Module):
    """
    Learned temporal attention pooling module.
    Takes sequence embeddings of shape (B, T, D) and computes
    learned attention weights a_t to produce an aggregated video representation (B, D).
    
    Formula:
        score_t = w^T * tanh(W_v * h_t + b)
        a = softmax(score, dim=1)
        v_video = sum_{t=1}^T (a_t * h_t)
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: embeddings of shape (B, T, D)
        Output:
            video_repr: (B, D)
            attention_weights: (B, T)
        """
        # Calculate raw attention scores: (B, T, 1)
        scores = self.attention_net(embeddings)

        # Normalize across temporal dimension: (B, T, 1)
        attention_weights = torch.softmax(scores, dim=1)

        # Compute weighted sum: (B, D)
        video_repr = torch.sum(attention_weights * embeddings, dim=1)

        return video_repr, attention_weights.squeeze(-1)
