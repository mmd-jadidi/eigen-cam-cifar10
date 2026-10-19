"""
Visualization utilities for Eigen-CAM heatmaps.

Provides functions to:
- Denormalize CIFAR-10 tensors for display
- Overlay CAM heatmaps on images using a colormap
- Display grids of images with their Eigen-CAM explanations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch


# ─── CIFAR-10 Constants ──────────────────────────────────────────────────────
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse CIFAR-10 normalization and convert to displayable numpy array.

    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        Numpy array of shape (H, W, C) or (B, H, W, C) with values clipped to [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().numpy()
    else:
        img = tensor

    if img.ndim == 3:
        # (C, H, W) → (H, W, C)
        img = img.transpose(1, 2, 0)
        img = img * CIFAR10_STD + CIFAR10_MEAN
    elif img.ndim == 4:
        # (B, C, H, W) → (B, H, W, C)
        img = img.transpose(0, 2, 3, 1)
        img = img * CIFAR10_STD + CIFAR10_MEAN

    return np.clip(img, 0, 1)


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a CAM heatmap on an image using a matplotlib colormap.

    Args:
        image: Original image as numpy array (H, W, C) in [0, 1].
        cam: CAM array (H, W) in [0, 1].
        alpha: Blending factor (0 = only image, 1 = only heatmap).
        colormap: Matplotlib colormap name.

    Returns:
        Blended image as numpy array (H, W, C) in [0, 1].
    """
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # (H, W, 3), drop alpha channel
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)


def show_eigen_cam_grid(
    images: list[np.ndarray],
    cams: list[np.ndarray],
    labels: list[str],
    predictions: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 8),
):
    """
    Display a grid with original images, heatmaps, and overlays.

    For each sample, shows 3 columns:
        1. Original image with true label
        2. Eigen-CAM heatmap
        3. Overlay with predicted label

    Args:
        images: List of denormalized images (H, W, C).
        cams: List of CAM arrays (H, W) in [0, 1].
        labels: List of ground-truth class names.
        predictions: List of predicted class names.
        save_path: If provided, save the figure to this path.
        figsize: Figure size.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=figsize)

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"True: {labels[i]}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Eigen-CAM heatmap
        axes[i, 1].imshow(cams[i], cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title("Eigen-CAM", fontsize=10)
        axes[i, 1].axis("off")

        # Overlay
        overlay = overlay_heatmap(images[i], cams[i])
        correct = labels[i] == predictions[i]
        color = "green" if correct else "red"
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Pred: {predictions[i]}", fontsize=10, color=color, fontweight="bold")
        axes[i, 2].axis("off")

    # Column headers
    axes[0, 0].set_title(f"Original\nTrue: {labels[0]}", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Eigen-CAM\nHeatmap", fontsize=10, fontweight="bold")
    color = "green" if labels[0] == predictions[0] else "red"
    axes[0, 2].set_title(f"Overlay\nPred: {predictions[0]}", fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def show_layer_comparison(
    image: np.ndarray,
    cams_dict: dict[str, np.ndarray],
    title: str = "Eigen-CAM Across Layers",
    save_path: str | None = None,
):
    """
    Compare Eigen-CAM outputs from different convolutional layers.

    Args:
        image: Denormalized original image (H, W, C).
        cams_dict: Dict mapping layer name → CAM array (H, W).
        title: Figure title.
        save_path: If provided, save the figure.
    """
    n_layers = len(cams_dict)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))

    # Show original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Show overlay for each layer
    for idx, (name, cam) in enumerate(cams_dict.items()):
        overlay = overlay_heatmap(image, cam, alpha=0.5)
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(name, fontsize=10, fontweight="bold")
        axes[idx + 1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()
