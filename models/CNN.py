import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightSymmetry(nn.Module):
    def forward(self, weight):
        return weight


class HorizontalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.5 * (weight + torch.flip(weight, dims=[3]))


class VerticalFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.5 * (weight + torch.flip(weight, dims=[2]))


class HVFlipSymmetry(WeightSymmetry):
    def forward(self, weight):
        return 0.25 * (
            weight +
            torch.flip(weight, dims=[2]) +
            torch.flip(weight, dims=[3]) +
            torch.flip(weight, dims=[2, 3])
        )


class Rot90Symmetry(WeightSymmetry):
    def forward(self, weight):
        rot90 = lambda w, k: torch.rot90(w, k=k, dims=[2, 3])
        return 0.25 * (weight + rot90(weight, 1) + rot90(weight, 2) + rot90(weight, 3))

SYMMETRY_CLASSES = {
    'vanilla': WeightSymmetry,
    'hflip': HorizontalFlipSymmetry,
    'vflip': VerticalFlipSymmetry,
    'hvflip': HVFlipSymmetry,
    'rot90': Rot90Symmetry
}


class SymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, symmetry='none', **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True, **kwargs)
        self.symmetry = SYMMETRY_CLASSES[symmetry]()  # instantiate symmetry handler

    def forward(self, x):
        weight = self.symmetry(self.conv.weight)
        return F.conv2d(
            x, weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )


class CNN(nn.Module):
    def __init__(self, symmetry=None):
        super().__init__()
        if symmetry is None:
            symmetry = ['vanilla'] * 4
        if isinstance(symmetry, str):
            symmetry = [symmetry] * 4

        self.conv1 = SymmetricConv2d(3, 64, kernel_size=3, padding=1, symmetry=symmetry[0])
        self.conv2 = SymmetricConv2d(64, 128, kernel_size=3, padding=1, symmetry=symmetry[1])
        self.conv3 = SymmetricConv2d(128, 256, kernel_size=3, padding=1, symmetry=symmetry[2])
        self.conv4 = SymmetricConv2d(256, 256, kernel_size=3, padding=1, symmetry=symmetry[3])

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
