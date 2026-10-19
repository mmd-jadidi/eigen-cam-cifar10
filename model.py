"""
Small CNN model for CIFAR-10 image classification.

Architecture:
    3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
    followed by adaptive average pooling and a linear classifier.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A compact CNN designed for CIFAR-10 (32x32 RGB images, 10 classes).

    Layer structure:
        Block 1: 3 → 32 channels, 3x3 conv, output 16x16
        Block 2: 32 → 64 channels, 3x3 conv, output 8x8
        Block 3: 64 → 128 channels, 3x3 conv, output 4x4
        Classifier: AdaptiveAvgPool → Flatten → FC(128, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3 — this is the target layer for Eigen-CAM
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_cam_target_layer(self) -> nn.Module:
        """Return the last convolutional layer for Eigen-CAM."""
        # features[8] is the Conv2d(64, 128, 3, padding=1) in Block 3
        return self.features[8]


if __name__ == "__main__":
    model = SmallCNN()
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print(f"Model output shape: {out.shape}")  # (1, 10)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
