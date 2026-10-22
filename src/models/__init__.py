"""
Video-Level Deepfake Detection Neural Architectures
"""
from .xception_backbone import XceptionFeatureExtractor
from .temporal_attention import TemporalAttentionPooling
from .video_classifier import TemporalDeepfakeDetector

__all__ = [
    "XceptionFeatureExtractor",
    "TemporalAttentionPooling",
    "TemporalDeepfakeDetector",
]
