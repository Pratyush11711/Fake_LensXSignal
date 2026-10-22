import torch
import torch.nn as nn
import torchvision.models as models


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, reps: int = 2, stride: int = 1, start_with_relu: bool = True):
        super().__init__()
        self.skip = None
        if out_filters != in_filters or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters),
            )

        rep = []
        filters = in_filters
        for i in range(reps):
            if start_with_relu or i > 0:
                rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, out_filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride=stride, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        return x + skip


class XceptionFeatureExtractor(nn.Module):
    """
    Xception Visual Backbone for facial artifact feature extraction.
    Takes image sequence tensor of shape (B, T, 3, H, W) and extracts
    frame-level feature vectors of shape (B, T, embedding_dim).
    """

    def __init__(self, embedding_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Entry flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False),
            XceptionBlock(128, 256, reps=2, stride=2),
            XceptionBlock(256, 728, reps=2, stride=2),
        )

        # Middle flow (8 residual blocks)
        middle_blocks = [XceptionBlock(728, 728, reps=3, stride=1) for _ in range(8)]
        self.middle_flow = nn.Sequential(*middle_blocks)

        # Exit flow
        self.exit_flow = nn.Sequential(
            XceptionBlock(728, 1024, reps=2, stride=2),
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, self.embedding_dim, 3, 1, 1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, 3, H, W) or (B*T, 3, H, W)
        Output: (B, T, embedding_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = 1
            x_flat = x

        feat = self.entry_flow(x_flat)
        feat = self.middle_flow(feat)
        feat = self.exit_flow(feat)  # (B*T, embedding_dim, 1, 1)

        embeddings = feat.view(B, T, self.embedding_dim)
        return embeddings
