# Standard library imports
import os
import sys
import inspect
from argparse import ArgumentParser

# Third-party imports
import numpy as np
import torch
from tabulate import tabulate
from PIL import Image
from tqdm import tqdm

# Local imports
from tools.evaluation import compute_pose_error, pose_auc, estimate_pose
from PMatch.PMatch.models import PMatch

# Constants
MIN_IMAGE_DIM = 480
NUM_MATCHES = 5000
NUM_ITERATIONS = 5
CONFIDENCE = 0.99999
POSE_THRESHOLDS = [5, 10, 20]
PIXEL_OFFSET = 0.5
NORM_THRESHOLD_FACTOR = 0.8
DEFAULT_ERROR = 90  # Default error value when pose estimation fails

# Add project root to system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

class ScanNetBenchmark:
    """Benchmark for evaluating PMatch performance on ScanNet dataset."""

    def __init__(self, data_root="data/scannet") -> None:
        """Initialize the benchmark with specified data root."""
        self.data_root = data_root

    def _load_intrinsics(self, scene_name):
        """Load camera intrinsics from file."""
        intrinsics_path = os.path.join(
            self.data_root,
            "scans_test",
            scene_name,
            "intrinsic",
            "intrinsic_color.txt"
        )
        with open(intrinsics_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            return np.stack([
                np.array([float(i) for i in line.split()])
                for line in lines
            ])

    def _get_image_path(self, scene_name, frame_id):
        """Construct path to image file."""
        return os.path.join(
            self.data_root,
            "scans_test",
            scene_name,
            "color",
            f"{frame_id}.jpg"
        )

    def _process_image_and_intrinsics(self, image_path, K):
        """Process image and its intrinsics matrix."""
        image = Image.open(image_path)
        width, height = image.size
        scale = MIN_IMAGE_DIM / min(width, height)
        new_width, new_height = scale * width, scale * height
        K_scaled = K * scale
        return image, new_width, new_height, K_scaled

    def _process_keypoints(self, matches, width, height):
        """Process keypoints from matches."""
        kpts = matches[:, :2] if matches.shape[1] == 2 else matches[:, 2:]
        return np.stack((
            width * (kpts[:, 0] + 1) / 2 - PIXEL_OFFSET,
            height * (kpts[:, 1] + 1) / 2 - PIXEL_OFFSET
        ), axis=-1)

    def _estimate_relative_pose(self, kpts1, kpts2, K1, K2):
        """Estimate relative pose from keypoints."""
        norm_threshold = NORM_THRESHOLD_FACTOR / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
        R_est, t_est, _ = estimate_pose(kpts1, kpts2, K1, K2, norm_threshold, conf=CONFIDENCE)
        return np.concatenate((R_est, t_est), axis=-1)

    @torch.no_grad()
    def benchmark(self, model):
        """Run benchmark evaluation on the dataset."""
        # Load test data
        test_data = np.load(os.path.join(self.data_root, "test.npz"))
        pairs, rel_pose = test_data["name"], test_data["rel_pose"]
        tot_e_t, tot_e_R, tot_e_pose = [], [], []

        # Set random seed for reproducibility
        np.random.seed(0)
        pair_indices = np.random.choice(len(pairs), size=len(pairs), replace=False)
        for pairind in tqdm(pair_indices):
            # Get scene information
            scene = pairs[pairind]
            scene_name = f"scene0{scene[0]}_00"
            
            # Get image paths and load images
            im1_path = self._get_image_path(scene_name, scene[2])
            im2_path = self._get_image_path(scene_name, scene[3])
            
            # Get ground truth pose
            T_gt = rel_pose[pairind].reshape(3, 4)
            R, t = T_gt[:3, :3], T_gt[:3, 3]
            
            # Load and process camera intrinsics
            K = self._load_intrinsics(scene_name)
            im1, w1, h1, K1 = self._process_image_and_intrinsics(im1_path, K.copy())
            im2, w2, h2, K2 = self._process_image_and_intrinsics(im2_path, K.copy())
            
            # Match features
            dense_matches, dense_certainty = model.match(im1, im2)
            sparse_matches, _ = model.sample(dense_matches, dense_certainty, NUM_MATCHES)
            
            # Process keypoints
            kpts1 = self._process_keypoints(sparse_matches[:, :2], w1, h1)
            kpts2 = self._process_keypoints(sparse_matches[:, 2:], w2, h2)
            
            # Perform multiple iterations with random permutations
            for _ in range(NUM_ITERATIONS):
                # Shuffle keypoints
                shuffling = np.random.permutation(len(kpts1))
                kpts1_shuffled = kpts1[shuffling]
                kpts2_shuffled = kpts2[shuffling]
                
                # Estimate pose and compute errors
                try:
                    T1_to_2_est = self._estimate_relative_pose(kpts1_shuffled, kpts2_shuffled, K1, K2)
                    if T1_to_2_est is None:
                        # Handle case where pose estimation returned None (insufficient matches)
                        e_t = e_R = DEFAULT_ERROR
                    else:
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                except (ValueError, np.linalg.LinAlgError) as e:
                    # Handle specific numerical errors that might occur during pose estimation
                    print(f"Warning: Pose estimation failed for scene {scene_name}, "
                          f"frames {scene[2]}-{scene[3]}: {str(e)}")
                    e_t = e_R = DEFAULT_ERROR
                except Exception as e:
                    # Handle any unexpected errors
                    print(f"Unexpected error in pose estimation for scene {scene_name}, "
                          f"frames {scene[2]}-{scene[3]}: {str(e)}")
                    e_t = e_R = DEFAULT_ERROR

                # Calculate final pose error
                e_pose = max(e_t, e_R)
                
                # Store errors
                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)

        # Calculate metrics
        tot_e_pose = np.array(tot_e_pose)
        auc = pose_auc(tot_e_pose, POSE_THRESHOLDS)
        
        # Calculate accuracies at different thresholds
        accuracies = [(tot_e_pose < threshold).mean() for threshold in [5, 10, 15, 20]]
        acc_5, acc_10, acc_15, acc_20 = accuracies
        
        # Calculate mean average precision
        result = {
            "auc_5": auc[0] * 100,
            "auc_10": auc[1] * 100,
            "auc_20": auc[2] * 100,
            "map_5": acc_5 * 100,
            "map_10": np.mean([acc_5, acc_10]) * 100,
            "map_20": np.mean([acc_5, acc_10, acc_15, acc_20]) * 100
        }
        
        return result

def setup_model(checkpoint_path):
    """Set up and configure the PMatch model."""
    model = PMatch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    
    # Configure model parameters
    model.cuda()
    model.eval()
    model.symmetric = True
    model.h_resized = 480
    model.w_resized = 640
    model.upsample_preds = False
    model.sample_mode = "threshold_balanced"
    model.factor = 0.5
    model.sample_thresh = 0.10
    
    return model

def main():
    """Main function to run the benchmark."""
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="/home/ubuntu/disk5/PMatchRelease/checkpoints/pmatch_scannet.pth"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ubuntu/disk5/TwoViewBenchmark/scannet_test_1500"
    )
    args, _ = parser.parse_known_args()

    # Set up model and run benchmark
    model = setup_model(args.checkpoints)
    benchmark = ScanNetBenchmark(args.data_path)
    result = benchmark.benchmark(model)
    
    # Display results
    print(tabulate(
        result.items(),
        headers=['Metric', 'Scores'],
        tablefmt='fancy_grid',
        floatfmt=".2f",
        numalign="left"
    ))

if __name__ == "__main__":
    main()
