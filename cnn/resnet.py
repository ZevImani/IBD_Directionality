"""ResNet for single-channel event images, regressing a 2-vector (cos theta, sin theta).

Based on https://github.com/JayPatwardhan/ResNet-PyTorch (Zev's modified ResNet50 from
Proton64_Reco_Model, now with a configurable stem and the `norm` option actually propagated).

stem:
  small    -- 3x3 stride-1 conv, no max-pool (CIFAR style). Use for small images (the
              displacement histograms are 5x5 at 150 mm pitch, 9x9 at 50 mm, 81x81 at 5 mm).
  imagenet -- 7x7 stride-2 conv + max-pool (the original proton64 model, /4 downsampling
              before layer1). Only sensible for large images (>~ 100 px, e.g. 1 mm pitch).
"""
import torch.nn as nn


def _norm_layer(norm):
    return nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, norm="batch"):
        super().__init__()
        Norm = _norm_layer(norm)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = Norm(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = Norm(out_channels * self.expansion)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        return self.relu(x + identity)


class Block(nn.Module):
    """Basic 2-conv residual block (ResNet18/34)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, norm="batch"):
        super().__init__()
        Norm = _norm_layer(norm)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = Norm(out_channels)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        return self.relu(x + identity)


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=2, num_channels=1, norm="batch", stem="small"):
        super().__init__()
        self.in_channels = 64
        self.norm = norm
        Norm = _norm_layer(norm)
        if stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                Norm(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif stem == "small":
            self.stem = nn.Sequential(
                nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                Norm(64), nn.ReLU())
        else:
            raise ValueError(f"stem must be small|imagenet, got {stem}")

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                _norm_layer(self.norm)(planes * ResBlock.expansion))
        layers = [ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, norm=self.norm)]
        self.in_channels = planes * ResBlock.expansion
        layers += [ResBlock(self.in_channels, planes, norm=self.norm) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)


ARCHS = {
    "resnet18": (Block, [2, 2, 2, 2]),
    "resnet34": (Block, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
    "resnet152": (Bottleneck, [3, 8, 36, 3]),
}


def build_model(arch="resnet50", num_classes=2, channels=1, norm="batch", stem="small"):
    block, layers = ARCHS[arch]
    return ResNet(block, layers, num_classes, channels, norm, stem)
