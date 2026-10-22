import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepfakeSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading video face crop sequences from metadata jsonl.
    Yields (sequence_tensor, label, video_id).
    Sequence tensor shape: (T, 3, H, W).
    """

    def __init__(
        self,
        metadata_path: str,
        sequence_length: int = 16,
        transform: Optional[Callable] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples: List[Dict] = []

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]
        frame_paths = item["frame_paths"]
        label = float(item["label"])  # 0.0 for REAL, 1.0 for FAKE
        video_id = item["video_id"]

        total_available = len(frame_paths)
        if total_available == 0:
            # Fallback zero tensor if empty
            return torch.zeros((self.sequence_length, 3, 256, 256)), torch.tensor([label], dtype=torch.float32), video_id

        # Uniform temporal striding
        indices = np.linspace(0, total_available - 1, self.sequence_length, dtype=int)
        sampled_paths = [frame_paths[i] for i in indices]

        frames = []
        for p in sampled_paths:
            img = cv2.imread(p)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

            if self.transform is not None:
                tensor = self.transform(img_rgb)
            else:
                tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

            frames.append(tensor)

        # Stack into (T, 3, H, W)
        sequence_tensor = torch.stack(frames, dim=0)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return sequence_tensor, label_tensor, video_id
