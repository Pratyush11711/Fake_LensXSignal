"""
detectors/lipforensics_detector.py

Isolated research detector wrapper for LipForensics (CVPR 2021).
Pretrained checkpoint: lipforensics_ff.pth (trained on FaceForensics++ c23).

Architecture:
- 3D Temporal Convolutional Frontend (Conv3d + BatchNorm3d + PReLU + MaxPool3d)
- 2D Spatial Trunk (ResNet-18 with PReLU activations)
- Temporal Backend: Multi-Scale Multi-Branch Temporal Convolutional Network (MS-TCN)

Input:
- Grayscale mouth ROI sequence tensor of shape (B, 1, T, 88, 88) where T=25 frames
- Normalized with mean=0.421, std=0.165

Output:
- Scalar logit per sequence (higher = synthetic / fake)
- Fake probability: sigmoid(logit)
"""

import math
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. TEMPORAL CONVOLUTIONAL NETWORK (MS-TCN) MODULES
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int, symm_chomp: bool = True):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2 : -self.chomp_size // 2].contiguous()
        else:
            return x[:, :, : -self.chomp_size].contiguous()


class ConvBatchChompRelu(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        relu_type: str = "prelu",
        dwpw: bool = False,
    ):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    n_inputs,
                    n_inputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=n_inputs,
                    bias=False,
                ),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == "prelu" else nn.ReLU(inplace=True),
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


class MultibranchTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_sizes: List[int],
        stride: int,
        dilation: int,
        padding: List[int],
        dropout: float = 0.2,
        relu_type: str = "prelu",
        dwpw: bool = False,
    ):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels must be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(
                n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw
            )
            setattr(self, f"cbcr0_{k_idx}", cbcr)
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(
                n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw
            )
            setattr(self, f"cbcr1_{k_idx}", cbcr)
        self.dropout1 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs // self.num_kernels) != n_outputs else None

        if relu_type == "relu":
            self.relu_final = nn.ReLU()
        elif relu_type == "prelu":
            self.relu_final = nn.PReLU(num_parameters=n_outputs)
        else:
            raise ValueError(f"Unsupported relu_type: {relu_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, f"cbcr0_{k_idx}")
            outputs.append(branch_convs(x))
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0(out0)

        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, f"cbcr1_{k_idx}")
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        tcn_options: Dict[str, Any],
        dropout: float = 0.2,
        relu_type: str = "prelu",
        dwpw: bool = False,
    ):
        super(MultibranchTemporalConvNet, self).__init__()
        self.ksizes = tcn_options["kernel_size"]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in self.ksizes]
            layers.append(
                MultibranchTemporalBlock(
                    in_channels,
                    out_channels,
                    self.ksizes,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    relu_type=relu_type,
                    dwpw=dwpw,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _average_batch(x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        num_classes: int,
        tcn_options: Dict[str, Any],
        dropout: float,
        relu_type: str,
        dwpw: bool = False,
    ):
        super(MultiscaleMultibranchTCN, self).__init__()
        self.kernel_sizes = tcn_options["kernel_size"]
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)
        self.consensus_func = _average_batch

    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.mb_ms_tcn(x)
        out = self.consensus_func(out, lengths)
        return self.tcn_output(out)


# ---------------------------------------------------------------------------
# 2. RESNET SPATIAL TRUNK
# ---------------------------------------------------------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(inplanes: int, outplanes: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        relu_type: str = "prelu",
    ):
        super(BasicBlock, self).__init__()
        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type, layers: List[int], relu_type: str = "prelu"):
        self.inplanes = 64
        self.relu_type = relu_type
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_basic_block(
                inplanes=self.inplanes, outplanes=planes * block.expansion, stride=stride
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# ---------------------------------------------------------------------------
# 3. COMPLETE LIPREADING ARCHITECTURE (3D FRONTEND + RESNET + MS-TCN)
# ---------------------------------------------------------------------------

def reshape_tensor(x: torch.Tensor) -> torch.Tensor:
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Lipreading(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 1,
        relu_type: str = "prelu",
        tcn_options: Optional[Dict[str, Any]] = None,
    ):
        super(Lipreading, self).__init__()
        if tcn_options is None:
            tcn_options = {
                "num_layers": 4,
                "kernel_size": [3, 5, 7],
                "dropout": 0.2,
                "dwpw": False,
                "width_mult": 1,
            }

        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        num_channels = [hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]] * tcn_options["num_layers"]
        self.tcn = MultiscaleMultibranchTCN(
            input_size=self.backend_out,
            num_channels=num_channels,
            num_classes=num_classes,
            tcn_options=tcn_options,
            dropout=tcn_options["dropout"],
            relu_type=relu_type,
            dwpw=tcn_options["dwpw"],
        )

    def forward(self, x: torch.Tensor, lengths: Optional[List[int]] = None) -> torch.Tensor:
        if lengths is None:
            lengths = [x.shape[2]] * x.shape[0]

        x = self.frontend3D(x)
        t_new = x.shape[2]
        x = reshape_tensor(x)
        x = self.trunk(x)
        x = x.view(-1, t_new, x.size(1))
        return self.tcn(x, lengths)


# ---------------------------------------------------------------------------
# 4. LIPFORENSICS DETECTOR WRAPPER
# ---------------------------------------------------------------------------

_MODEL_SINGLETON: Optional[Lipreading] = None
_CURRENT_DEVICE: Optional[torch.device] = None


def find_checkpoint_path(custom_path: Optional[str] = None) -> Path:
    """Locates the verified lipforensics_ff.pth checkpoint."""
    if custom_path and os.path.exists(custom_path) and Path(custom_path).stat().st_size > 10_000_000:
        return Path(custom_path)

    import tempfile
    base_dir = Path(__file__).resolve().parent.parent
    candidate_locations = [
        base_dir / "models" / "weights" / "lipforensics_ff.pth",
        base_dir / "models" / "lipforensics_ff.pth",
        Path("models/weights/lipforensics_ff.pth"),
        Path("models/lipforensics_ff.pth"),
        Path(tempfile.gettempdir()) / "models" / "weights" / "lipforensics_ff.pth",
        Path(tempfile.gettempdir()) / "lipforensics_ff.pth",
        base_dir / "vendor" / "lipforensics_ff.pth",
    ]

    for p_cand in candidate_locations:
        if p_cand.exists() and p_cand.stat().st_size > 10_000_000:
            return p_cand

    # Serverless runtime fallback: download and verify into /tmp if not yet present
    target_tmp = Path(tempfile.gettempdir()) / "models" / "weights" / "lipforensics_ff.pth"
    try:
        from scripts.download_weights import download_lipforensics_checkpoint
        download_lipforensics_checkpoint(target_tmp)
        if target_tmp.exists() and target_tmp.stat().st_size > 10_000_000:
            return target_tmp
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate lipforensics_ff.pth. Checked locations: "
        + ", ".join(str(p_cand) for p in candidate_locations)
    )


def get_lipforensics_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Lipreading:
    """Loads and caches the singleton LipForensics model."""
    global _MODEL_SINGLETON, _CURRENT_DEVICE

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    if _MODEL_SINGLETON is not None and _CURRENT_DEVICE == target_device:
        return _MODEL_SINGLETON

    ckpt_file = find_checkpoint_path(checkpoint_path)
    model = Lipreading(
        hidden_dim=256,
        num_classes=1,
        relu_type="prelu",
        tcn_options={
            "num_layers": 4,
            "kernel_size": [3, 5, 7],
            "dropout": 0.2,
            "dwpw": False,
            "width_mult": 1,
        },
    )

    checkpoint = torch.load(str(ckpt_file), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device)
    model.eval()

    _MODEL_SINGLETON = model
    _CURRENT_DEVICE = target_device
    return _MODEL_SINGLETON
