"""
Training script for the SmallCNN on CIFAR-10.

Usage:
    python train.py                     # train with defaults (10 epochs)
    python train.py --epochs 20         # train for 20 epochs
    python train.py --lr 0.005          # custom learning rate
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN


# ─── CIFAR-10 Statistics ─────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_dataloaders(batch_size: int = 128, data_dir: str = "./data"):
    """Create train and test dataloaders with standard CIFAR-10 augmentation."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def train(epochs: int = 10, lr: float = 0.01, batch_size: int = 128):
    """Full training loop with checkpointing."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SmallCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Ensure checkpoint directory exists
    os.makedirs("checkpoints", exist_ok=True)

    best_acc = 0.0
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {val_loss:8.4f} | {val_acc:6.2f}% | {elapsed:5.1f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, "checkpoints/best_model.pth")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to checkpoints/best_model.pth")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmallCNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
