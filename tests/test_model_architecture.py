import pytest
import torch

from src.models.temporal_attention import TemporalAttentionPooling
from src.models.video_classifier import TemporalDeepfakeDetector
from src.models.xception_backbone import XceptionFeatureExtractor


def test_xception_feature_extractor_shape():
    """Verify Xception extracts (B, T, 2048) embeddings from (B, T, 3, 256, 256)."""
    backbone = XceptionFeatureExtractor(embedding_dim=2048)
    x = torch.randn(2, 4, 3, 256, 256)
    embeddings = backbone(x)
    assert embeddings.shape == (2, 4, 2048)


def test_temporal_attention_pooling():
    """Verify temporal attention outputs (B, 2048) video vector and normalized weights (B, T)."""
    attention_module = TemporalAttentionPooling(input_dim=2048, hidden_dim=256)
    embeddings = torch.randn(2, 8, 2048)
    video_repr, weights = attention_module(embeddings)

    assert video_repr.shape == (2, 2048)
    assert weights.shape == (2, 8)

    # Attention weights must sum to 1.0 across the sequence
    sums = torch.sum(weights, dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_end_to_end_temporal_classifier_forward():
    """Verify end-to-end model forward pass produces valid logits, probabilities, and weights."""
    model = TemporalDeepfakeDetector(embedding_dim=2048, hidden_dim=256)
    model.eval()

    sequence_tensor = torch.randn(2, 8, 3, 256, 256)
    logits, probs, attn_weights = model(sequence_tensor)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert attn_weights.shape == (2, 8)

    # Probabilities must be bounded in [0.0, 1.0]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
