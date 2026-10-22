"""
detectors/gend_detector.py

Isolated wrapper for GenD (GenD_PE_L pretrained checkpoint).
Provides standalone frame-level deepfake inference returning the verified scalar fake logit.

Model: yermandy/GenD_PE_L
Backbone: vit_pe_core_large_patch14_336 (PerceptionEncoder EVA-02 Large)
Head: LinearProbe / LinearNorm (dim: 1024 -> 2 classes)
Outputs:
  Index 0: REAL
  Index 1: FAKE
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Union, List
import torch
import timm
from PIL import Image

# Import GenD model class dynamically to avoid namespace collisions with local `src`
_VENDOR_GEND_PATH = Path(__file__).resolve().parent.parent / "vendor" / "gend_src"
_MODELING_FILE = _VENDOR_GEND_PATH / "src" / "hf" / "modeling_gend.py"

if not _MODELING_FILE.exists():
    raise FileNotFoundError(f"GenD modeling file not found at: {_MODELING_FILE}")

_spec = importlib.util.spec_from_file_location("gend_hf_modeling", str(_MODELING_FILE))
_gend_module = importlib.util.module_from_spec(_spec)
sys.modules["gend_hf_modeling"] = _gend_module
_spec.loader.exec_module(_gend_module)

GenD = _gend_module.GenD

# Transformers compatibility attributes
if not hasattr(GenD, "all_tied_weights_keys"):
    GenD.all_tied_weights_keys = {}
if not hasattr(GenD, "_tied_weights_keys"):
    GenD._tied_weights_keys = []

_MODEL_SINGLETON: Optional[GenD] = None
_CURRENT_DEVICE: Optional[torch.device] = None
DEFAULT_MODEL_NAME = "yermandy/GenD_PE_L"


def get_gend_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None
) -> GenD:
    """
    Load or retrieve the cached singleton GenD model instance.
    """
    global _MODEL_SINGLETON, _CURRENT_DEVICE

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    if _MODEL_SINGLETON is None or _CURRENT_DEVICE != target_device:
        model = GenD.from_pretrained(model_name)
        model.eval()

        # Fix any meta-device transforms created during empty weight initialization
        if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "backbone"):
            try:
                data_config = timm.data.resolve_model_data_config(model.feature_extractor.backbone)
                data_config["input_size"] = (3, 224, 224)
                model.feature_extractor._preprocess = timm.data.create_transform(**data_config, is_training=False)
            except Exception:
                pass

        model.to(target_device)
        _MODEL_SINGLETON = model
        _CURRENT_DEVICE = target_device

    return _MODEL_SINGLETON


def predict_frame_logit(
    image: Image.Image,
    model: Optional[GenD] = None,
    device: Optional[Union[str, torch.device]] = None
) -> float:
    """
    Accepts a PIL Image, applies GenD's native preprocessing,
    runs inference under torch.no_grad(), and returns the verified scalar FAKE logit.

    Args:
        image: PIL Image to inspect
        model: Optional pre-loaded GenD model instance. If None, uses singleton.
        device: Optional device specification ('cpu', 'cuda', etc.)

    Returns:
        float: Scalar FAKE logit (index 1 of output tensor)
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    if model is None:
        model = get_gend_model(device=device)

    # Use the model's native preprocessing
    # PerceptionEncoder._preprocess maps PIL Image -> torch.Tensor of shape (3, 224, 224)
    tensor = model.feature_extractor.preprocess(image)

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Move to model's parameter device
    model_device = next(model.parameters()).device
    tensor = tensor.to(model_device)

    with torch.no_grad():
        logits = model(tensor)  # Shape: (1, 2)
        # Index 0 is REAL, Index 1 is FAKE
        fake_logit = float(logits[0, 1].item())

    return fake_logit


def predict_frame_prob(
    image: Image.Image,
    model: Optional[GenD] = None,
    device: Optional[Union[str, torch.device]] = None
) -> float:
    """
    Accepts a PIL Image and returns the softmax probability of the image being FAKE.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    if model is None:
        model = get_gend_model(device=device)

    tensor = model.feature_extractor.preprocess(image)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    model_device = next(model.parameters()).device
    tensor = tensor.to(model_device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        fake_prob = float(probs[0, 1].item())

    return fake_prob


def predict_batch_logits(
    images: List[Image.Image],
    model: Optional[GenD] = None,
    device: Optional[Union[str, torch.device]] = None
) -> List[float]:
    """
    Batch prediction helper returning a list of scalar FAKE logits.
    """
    if not images:
        return []

    if model is None:
        model = get_gend_model(device=device)

    rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    tensors = torch.stack([model.feature_extractor.preprocess(img) for img in rgb_images])

    model_device = next(model.parameters()).device
    tensors = tensors.to(model_device)

    with torch.no_grad():
        logits = model(tensors)  # Shape: (B, 2)
        fake_logits = [float(logits[i, 1].item()) for i in range(len(images))]

    return fake_logits
