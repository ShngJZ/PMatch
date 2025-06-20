from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F

# Constants
DEPTH_CONSISTENCY_THRESHOLD = 0.05
ZERO_DEPTH_EPSILON = 1e-4
@torch.no_grad()
def warp_kpts(
    kpts0: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    T_0to1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp keypoints from view 0 to view 1 using depth maps and camera parameters.
    
    This function performs the following steps:
    1. Unproject keypoints to 3D using depth from view 0
    2. Transform points using relative pose
    3. Project points to view 1
    4. Check visibility and depth consistency
    
    Adapted from LoFTR: https://github.com/zju3dv/LoFTR

    Args:
        kpts0: Keypoints in view 0, normalized to [-1,1] (shape: [N, L, 2])
        depth0: Depth map of view 0 (shape: [N, H, W])
        depth1: Depth map of view 1 (shape: [N, H, W])
        T_0to1: Transform from view 0 to 1 (shape: [N, 3, 4])
        K0: Intrinsic matrix of view 0 (shape: [N, 3, 3])
        K1: Intrinsic matrix of view 1 (shape: [N, 3, 3])

    Returns:
        Tuple containing:
        - valid_mask: Boolean mask for valid warped points (shape: [N, L])
        - warped_kpts: Warped keypoints in view 1, normalized to [-1,1] (shape: [N, L, 2])
    """
    # Get image dimensions and sample depths at keypoint locations
    n, h, w = depth0.shape
    kpts0_depth = F.grid_sample(
        depth0[:, None],
        kpts0[:, :, None],
        mode="bilinear",
        align_corners=False
    )[:, 0, :, 0]

    # Convert normalized coordinates to pixel coordinates
    kpts0_px = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2),
        dim=-1
    )

    # Create mask for valid depths
    nonzero_mask = kpts0_depth != 0

    # Unproject keypoints to 3D space
    kpts0_homogeneous = torch.cat(
        [kpts0_px, torch.ones_like(kpts0_px[:, :, [0]])],
        dim=-1
    ) * kpts0_depth[..., None]
    kpts0_cam = K0.inverse() @ kpts0_homogeneous.transpose(2, 1)

    # Apply rigid transformation
    kpts0_transformed = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]
    transformed_depth = kpts0_transformed[:, 2, :]

    # Project points to second view
    kpts1_homogeneous = (K1 @ kpts0_transformed).transpose(2, 1)
    kpts1_px = kpts1_homogeneous[:, :, :2] / (
        kpts1_homogeneous[:, :, [2]] + ZERO_DEPTH_EPSILON
    )

    # Check if points are within image bounds
    covisible_mask = (
        (kpts1_px[:, :, 0] > 0)
        * (kpts1_px[:, :, 0] < w - 1)
        * (kpts1_px[:, :, 1] > 0)
        * (kpts1_px[:, :, 1] < h - 1)
    )

    # Convert back to normalized coordinates
    kpts1_norm = torch.stack(
        (2 * kpts1_px[..., 0] / w - 1, 2 * kpts1_px[..., 1] / h - 1),
        dim=-1
    )

    # Check depth consistency
    warped_depth = F.grid_sample(
        depth1[:, None],
        kpts1_norm[:, :, None],
        mode="bilinear",
        align_corners=False
    )[:, 0, :, 0]
    
    depth_consistent = (
        (warped_depth - transformed_depth) / warped_depth
    ).abs() < DEPTH_CONSISTENCY_THRESHOLD

    # Combine all validity checks
    valid_mask = nonzero_mask * covisible_mask * depth_consistent

    return valid_mask, kpts1_norm

def coords_gridN(
    batch: int,
    ht: int,
    wd: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generate a normalized coordinate grid.

    Args:
        batch: Batch size
        ht: Grid height
        wd: Grid width
        device: Target device for the grid

    Returns:
        Coordinate grid of shape [B, 2, H, W] normalized to [-1, 1]
    """
    coords = torch.meshgrid(
        torch.linspace(-1 + 1/ht, 1 - 1/ht, ht, device=device),
        torch.linspace(-1 + 1/wd, 1 - 1/wd, wd, device=device),
    )
    
    return torch.stack(
        (coords[1], coords[0]),
        dim=0
    )[None].repeat(batch, 1, 1, 1)

def to_cuda(batch: Dict[str, Union[torch.Tensor, object]]) -> Dict[str, Union[torch.Tensor, object]]:
    """
    Move all torch.Tensor objects in a dictionary to CUDA device.

    Args:
        batch: Dictionary containing tensors to move to CUDA

    Returns:
        Dictionary with all tensors moved to CUDA device
    """
    return {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

def unnorm_coords_Numpystyle(
    coords: torch.Tensor,
    h: int,
    w: int
) -> torch.Tensor:
    """
    Convert normalized coordinates [-1, 1] to pixel coordinates [0, size].

    Args:
        coords: Normalized coordinates tensor of shape [B, 2, H, W]
        h: Target height in pixels
        w: Target width in pixels

    Returns:
        Pixel coordinates tensor of shape [B, 2, H, W]
    """
    coords_x, coords_y = torch.split(coords, 1, dim=1)
    coords_x = (coords_x + 1) / 2 * w
    coords_y = (coords_y + 1) / 2 * h
    return torch.cat([coords_x, coords_y], dim=1)
