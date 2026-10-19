"""
Eigen-CAM: Visual Explanations via Principal Components of Activations.

Eigen-CAM computes a class-agnostic saliency map by performing SVD on the
activation maps from a target convolutional layer. The first principal
component captures the most important spatial patterns the network attends to.

Reference:
    Muhammad, M. B., & Yeasin, M. (2020).
    "Eigen-CAM: Class Activation Map using Principal Components."
    arXiv:2008.00299
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EigenCAM:
    """
    Eigen-CAM explainability method.

    Generates a saliency heatmap by extracting the first principal component
    of the activation maps from a target convolutional layer via SVD.

    Args:
        model: A trained PyTorch CNN model.
        target_layer: The convolutional layer to extract activations from.

    Example:
        >>> model = SmallCNN()
        >>> eigen_cam = EigenCAM(model, model.get_cam_target_layer())
        >>> cam = eigen_cam.generate(input_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        """Register a forward hook to capture activations."""
        def hook_fn(module, input, output):
            self.activations = output.detach()

        self._hook = self.target_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        """Remove the registered hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Generate an Eigen-CAM saliency map for the given input.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_size: (H, W) to resize the CAM to. Defaults to input spatial size.

        Returns:
            cam: Numpy array of shape (H, W) with values in [0, 1].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward pass to trigger the hook and capture activations
        self.model.eval()
        _ = self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("No activations captured. Is the hook registered?")

        # activations shape: (1, C, H, W) → squeeze batch → (C, H, W)
        act = self.activations.squeeze(0)  # (C, H, W)
        C, H, W = act.shape

        # Reshape to 2D matrix: (C, H*W)
        act_2d = act.reshape(C, H * W).cpu().numpy()

        # ─── SVD ──────────────────────────────────────────────────────
        # U: (C, C), S: (min(C, H*W),), Vt: (H*W, H*W) or truncated
        # The first row of Vt (first right singular vector) captures
        # the dominant spatial pattern across all channels.
        _, _, Vt = np.linalg.svd(act_2d, full_matrices=False)

        # First principal component → reshape to spatial map
        cam = Vt[0, :]  # shape: (H*W,)
        cam = cam.reshape(H, W)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to target size (default: input image size)
        if target_size is None:
            target_size = (input_tensor.shape[2], input_tensor.shape[3])

        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_resized = F.interpolate(
            cam_tensor, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam_resized.squeeze().numpy()

        return cam

    @torch.no_grad()
    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """
        Generate Eigen-CAM maps for a batch of images.

        Args:
            input_tensor: Batch of images, shape (B, C, H, W).
            target_size: (H, W) to resize each CAM to.

        Returns:
            List of numpy arrays, each of shape (H, W) in [0, 1].
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        self.model.eval()
        _ = self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("No activations captured.")

        # activations shape: (B, C, H, W)
        batch_act = self.activations.cpu().numpy()
        B, C, H, W = batch_act.shape

        if target_size is None:
            target_size = (input_tensor.shape[2], input_tensor.shape[3])

        cams = []
        for i in range(B):
            act_2d = batch_act[i].reshape(C, H * W)  # (C, H*W)
            _, _, Vt = np.linalg.svd(act_2d, full_matrices=False)

            cam = Vt[0, :].reshape(H, W)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
            cam_resized = F.interpolate(
                cam_tensor, size=target_size, mode="bilinear", align_corners=False
            )
            cams.append(cam_resized.squeeze().numpy())

        return cams

    def __del__(self):
        self.remove_hook()


if __name__ == "__main__":
    from model import SmallCNN

    model = SmallCNN()
    eigen_cam = EigenCAM(model, model.get_cam_target_layer())

    # Generate CAM for a random input
    dummy = torch.randn(1, 3, 32, 32)
    cam = eigen_cam.generate(dummy)
    print(f"CAM shape: {cam.shape}, min: {cam.min():.3f}, max: {cam.max():.3f}")

    # Batch generate
    batch = torch.randn(4, 3, 32, 32)
    cams = eigen_cam.generate_batch(batch)
    print(f"Batch CAMs: {len(cams)} maps, each shape: {cams[0].shape}")

    eigen_cam.remove_hook()
    print("Eigen-CAM test passed!")
