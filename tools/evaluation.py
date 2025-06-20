from typing import Tuple, List, Optional, Union
import cv2
import torch
import numpy as np
from einops.einops import rearrange
from tools.tools import warp_kpts, unnorm_coords_Numpystyle

# Constants
ESSENTIAL_MAT_CONF = 0.99999
RECOVER_POSE_DIST = 1e9
AMBIGUITY_ANGLE = 180

def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Calculate the angular error between two rotation matrices.

    Args:
        R1: First rotation matrix (3x3)
        R2: Second rotation matrix (3x3)

    Returns:
        Angular error in degrees
    """
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # Handle numerical errors
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angular error between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angular error in degrees
    """
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    """
    Compute pose error metrics between estimated and ground truth poses.

    Args:
        T_0to1: Ground truth transformation matrix (4x4 or 3x4)
        R: Estimated rotation matrix (3x3)
        t: Estimated translation vector (3x1)

    Returns:
        Tuple of (translation error, rotation error) in degrees
    """
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, AMBIGUITY_ANGLE - error_t)  # Handle E matrix ambiguity
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors: np.ndarray, thresholds: List[float]) -> List[float]:
    """
    Compute Area Under the Curve (AUC) for pose errors at different thresholds.

    Args:
        errors: Array of pose errors
        thresholds: List of error thresholds to evaluate AUC at

    Returns:
        List of AUC values corresponding to each threshold
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    
    for threshold in thresholds:
        last_index = np.searchsorted(errors, threshold)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], threshold]
        aucs.append(np.trapz(r, x=e) / threshold)
    
    return aucs

def compute_relative_pose(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose between two camera positions.

    Args:
        R1: First rotation matrix (3x3)
        t1: First translation vector (3x1)
        R2: Second rotation matrix (3x3)
        t2: Second translation vector (3x1)

    Returns:
        Tuple of (relative rotation matrix, relative translation vector)
    """
    rots = R2 @ R1.T
    trans = -rots @ t1 + t2
    return rots, trans

def rotate_intrinsic(K: np.ndarray, n: int) -> np.ndarray:
    """
    Rotate camera intrinsic matrix by 90 degrees n times.

    Args:
        K: Camera intrinsic matrix (3x3)
        n: Number of 90-degree rotations to apply

    Returns:
        Rotated intrinsic matrix
    """
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K

def estimate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = ESSENTIAL_MAT_CONF
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimate relative pose between two views using matched keypoints.

    Args:
        kpts0: Keypoints in first image (Nx2)
        kpts1: Keypoints in second image (Nx2)
        K0: Intrinsic matrix of first camera (3x3)
        K1: Intrinsic matrix of second camera (3x3)
        norm_thresh: Threshold for normalized point distance
        conf: Confidence threshold for RANSAC (default: 0.99999)

    Returns:
        Tuple of (rotation matrix, translation vector, inlier mask) or None if estimation fails
    """
    if len(kpts0) < 5:
        return None

    # Normalize keypoints using inverse intrinsic matrices
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])
    kpts0_norm = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1_norm = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T

    # Find essential matrix
    E, mask = cv2.findEssentialMat(
        kpts0_norm,
        kpts1_norm,
        np.eye(3),
        threshold=norm_thresh,
        prob=conf,
        method=cv2.RANSAC
    )

    if E is None:
        return None

    # Recover pose from essential matrix
    best_num_inliers = 0
    ret = None

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E,
            kpts0_norm,
            kpts1_norm,
            np.eye(3),
            RECOVER_POSE_DIST,
            mask=mask
        )
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)

    return ret


# -- # Compute Error in EPE

def compute_flow_metrics(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    T_1to2: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    dense_matches: torch.Tensor
) -> torch.Tensor:
    """
    Compute optical flow metrics between two views.

    Args:
        depth1: Depth map of first view (BxHxW)
        depth2: Depth map of second view (BxHxW)
        T_1to2: Transformation matrix from view 1 to 2 (Bx4x4)
        K1: Intrinsic matrix of first camera (Bx3x3)
        K2: Intrinsic matrix of second camera (Bx3x3)
        dense_matches: Dense flow field (BxCxHxW)

    Returns:
        Tensor containing [EPE sum, #pixels<1px, #pixels<5px, #pixels<8px, #valid pixels]
    """
    # Rearrange dense matches for processing
    dense_matches = rearrange(dense_matches, "b d h w -> b h w d")
    b, h1, w1, d = dense_matches.shape

    # Generate normalized grid coordinates
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1/n, 1 - 1/n, n, device=dense_matches.device)
            for n in (b, h1, w1)
        ]
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(b, h1 * w1, 2)

    # Warp keypoints using depth and camera parameters
    mask, x2 = warp_kpts(
        x1_n.double(),
        depth1.double(),
        depth2.double(),
        T_1to2.double(),
        K1.double(),
        K2.double(),
    )
    mask_validation = mask.float().reshape(b, h1, w1) > 0
    x2 = x2.reshape(b, h1, w1, 2)

    # Convert to pixel coordinates
    dense_matches = unnorm_coords_Numpystyle(
        rearrange(dense_matches, "b h w d -> b d h w"),
        h=h1, w=w1
    )
    x2 = unnorm_coords_Numpystyle(
        rearrange(x2, "b h w d -> b d h w"),
        h=h1, w=w1
    )

    # Compute endpoint error
    epe = (dense_matches - x2).norm(dim=1)

    # Compute metrics
    evaluated_pixel_num = torch.sum(mask_validation)
    epe_sum = torch.sum(epe[mask_validation])
    px1 = torch.sum((epe < 1)[mask_validation])
    px5 = torch.sum((epe < 5)[mask_validation])
    px8 = torch.sum((epe < 8)[mask_validation])

    return torch.stack([epe_sum, px1, px5, px8, evaluated_pixel_num])
