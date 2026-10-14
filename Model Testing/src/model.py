"""
Step 3a - The model: a small CNN that reads a log-mel spectrogram like a picture.

Input : log-mel spectrogram  [batch, 64 mel bands, 101 frames]   (from features.py)
Output: 2 scores ("logits")  [batch, 2]
        cough detector: 0 = not a cough, 1 = cough
        dry/wet model : 0 = dry,         1 = wet

Structure
    normalise each spectrogram (mean 0, std 1)
    -> 4 conv blocks:  Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2
       channels 1 -> 16 -> 32 -> 64 -> 128, size 64x101 -> 32x50 -> 16x25 -> 8x12 -> 4x6
    -> global average pooling  (128 numbers: "which patterns appeared")
    -> dropout -> linear layer -> 2 logits

The network is split into a BACKBONE (the conv blocks, which learn general sound patterns)
and a HEAD (the last linear layer, which makes the decision). Transfer learning:
    1. train backbone + head on cough vs non-cough  (~18,500 training clips)
    2. keep the backbone, put a NEW head on it, fine-tune on dry vs wet (~1,400 clips)

IMPORTANT: this file is copied unchanged into the app (Final Project/app/model.py).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class Backbone(nn.Module):
    """Feature learner: spectrogram [B, 64, T] -> vector [B, 128]."""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = c_in

    @staticmethod
    def normalise(x):
        """Per-spectrogram standardisation, done inside the model so the app cannot forget it."""
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + 1e-5)

    def forward(self, x):
        x = self.normalise(x).unsqueeze(1)   # [B, 64, T] -> [B, 1, 64, T]  (1 "colour channel")
        x = self.blocks(x)                   # [B, 128, 4, T/16]
        return x.mean(dim=(2, 3))            # global average pooling -> [B, 128]


class CoughNet(nn.Module):
    def __init__(self, n_classes=2, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        self.backbone = Backbone(channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.out_dim, n_classes)

    def forward(self, x):
        """x: [B, n_mels, T] log-mel -> logits [B, n_classes]."""
        return self.head(self.dropout(self.backbone(x)))

    @torch.no_grad()
    def predict_proba(self, x):
        """Probabilities (softmax of the logits). Used by evaluation and by the app."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)

    def replace_head(self, n_classes=2):
        """Transfer learning: keep the learned backbone, start a fresh decision layer."""
        self.head = nn.Linear(self.backbone.out_dim, n_classes)

    def freeze_backbone(self, frozen=True):
        """Optionally train only the head first, so the new head can't wreck the pretrained backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = not frozen


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import time
    m = CoughNet()
    print(m)
    print(f"parameters: {count_parameters(m):,}")
    x = torch.randn(8, 64, 101)
    print("input", tuple(x.shape), "-> output", tuple(m(x).shape))
    m.eval()
    one = torch.randn(1, 64, 101)
    with torch.no_grad():
        for _ in range(5):
            m(one)
        t = time.perf_counter()
        for _ in range(100):
            m(one)
    print(f"CPU inference: {(time.perf_counter() - t) * 10:.2f} ms per 1-second clip")
