from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .xception_backbone import XceptionFeatureExtractor
from .temporal_attention import TemporalAttentionPooling


class TemporalDeepfakeDetector(nn.Module):
    """
    End-to-End Video-Level Deepfake Detector:
    Input Sequence: (B, T, 3, 256, 256)
    -> Xception Visual Backbone -> Frame Embeddings (B, T, 2048)
    -> Learned Temporal Attention Pooling -> Video Embedding (B, 2048) + Weights (B, T)
    -> Classifier Head -> Video Logit (B, 1) & Probability (B, 1)
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = XceptionFeatureExtractor(embedding_dim=embedding_dim)
        self.temporal_attention = TemporalAttentionPooling(input_dim=embedding_dim, hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: (B, T, 3, H, W)
        Returns:
            logits: (B, 1) raw logit output
            probabilities: (B, 1) sigmoid normalized probability
            attention_weights: (B, T) learned attention per frame
        """
        # 1. Visual Feature Extraction: (B, T, D)
        embeddings = self.backbone(x)

        # 2. Learned Temporal Attention Pooling: (B, D), (B, T)
        video_repr, attention_weights = self.temporal_attention(embeddings)

        # 3. Classification Head: (B, 1)
        logits = self.classifier(video_repr)
        probabilities = torch.sigmoid(logits)

        return logits, probabilities, attention_weights
