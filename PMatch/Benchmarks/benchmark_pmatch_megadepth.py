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
from tools.evaluation import compute_pose_error, pose_auc, compute_relative_pose, estimate_pose
from PMatch.PMatch.models import PMatch

# Constants
MAX_IMAGE_DIM = 1200
NUM_MATCHES = 5000
NUM_ITERATIONS = 5
CONFIDENCE = 0.99999
POSE_THRESHOLDS = [5, 10, 20]

# Add project root to system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

class Megadepth1500Benchmark:
    """Benchmark for evaluating PMatch performance on MegaDepth dataset."""
    
    DEFAULT_SCENES = [
        "0015_0.1_0.3.npz",
        "0015_0.3_0.5.npz",
        "0022_0.1_0.3.npz",
        "0022_0.3_0.5.npz",
        "0022_0.5_0.7.npz",
    ]

    def __init__(self, data_root="data/megadepth", scene_names=None) -> None:
        """Initialize the benchmark with specified data root and scene names."""
        self.scene_names = scene_names if scene_names is not None else self.DEFAULT_SCENES
        self.scenes = [
            np.load(f"{data_root}/scene_info/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def _process_image_and_intrinsics(self, image_path, K):
        """Process image and its intrinsics matrix."""
        image = Image.open(image_path)
        width, height = image.size
        scale = MAX_IMAGE_DIM / max(width, height)
        new_width, new_height = scale * width, scale * height
        K_scaled = K.copy()
        K_scaled[:2] = K_scaled[:2] * scale
        return image, new_width, new_height, K_scaled

    def _process_keypoints(self, matches, width, height):
        """Process keypoints from matches."""
        kpts = matches[:, :2] if matches.shape[1] == 2 else matches[:, 2:]
        return np.stack((
            width * (kpts[:, 0] + 1) / 2,
            height * (kpts[:, 1] + 1) / 2
        ), axis=-1)

    def _estimate_relative_pose(self, kpts1, kpts2, K1, K2):
        """Estimate relative pose from keypoints."""
        norm_threshold = 0.5 / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
        R_est, t_est, _ = estimate_pose(kpts1, kpts2, K1, K2, norm_threshold, conf=CONFIDENCE)
        return np.concatenate((R_est, t_est), axis=-1)

    @torch.no_grad()
    def benchmark(self, model):
        """Run benchmark evaluation on the dataset."""
        tot_e_t, tot_e_R, tot_e_pose = [], [], []
        
        for scene in self.scenes:
            pairs = scene["pair_infos"]
            intrinsics = scene["intrinsics"]
            poses = scene["poses"]
            im_paths = scene["image_paths"]
            
            for pairind in tqdm(range(len(pairs)), disable=False):
                idx1, idx2 = pairs[pairind][0]
                
                # Get camera parameters
                K1, T1 = intrinsics[idx1].copy(), poses[idx1].copy()
                K2, T2 = intrinsics[idx2].copy(), poses[idx2].copy()
                R1, t1 = T1[:3, :3], T1[:3, 3]
                R2, t2 = T2[:3, :3], T2[:3, 3]
                R, t = compute_relative_pose(R1, t1, R2, t2)

                # Process images and their intrinsics
                im1_path = f"{self.data_root}/{im_paths[idx1]}"
                im2_path = f"{self.data_root}/{im_paths[idx2]}"
                im1, w1, h1, K1 = self._process_image_and_intrinsics(im1_path, K1)
                im2, w2, h2, K2 = self._process_image_and_intrinsics(im2_path, K2)

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
                    T1_to_2_est = self._estimate_relative_pose(kpts1_shuffled, kpts2_shuffled, K1, K2)
                    e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
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
    model.upsample_preds = True
    model.symmetric = True
    model.h_resized = 664
    model.w_resized = 872
    model.sample_mode = "threshold_balanced"
    model.factor = 0.8
    model.sample_thresh = 0.04
    
    return model

def main():
    """Main function to run the benchmark."""
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="/home/ubuntu/disk5/PMatchRelease/checkpoints/pmatch_mega.pth"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ubuntu/disk5/TwoViewBenchmark/megadepth_test_1500"
    )
    args, _ = parser.parse_known_args()

    # Set up model and run benchmark
    model = setup_model(args.checkpoints)
    benchmark = Megadepth1500Benchmark(args.data_path)
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
