"""
Demo script for PMatch that processes two RGB images from assets folder.
Based on benchmark_pmatch_megadepth.py but simplified for direct image input.

This script uses:
- Input: assets/img1.jpg and assets/img2.jpg
- Output: assets/demo.png
"""


# Standard library imports
import os
import sys
import inspect
from argparse import ArgumentParser

# Third-party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Local imports
from PMatch.PMatch.models import PMatch
from tools.tools import unnorm_coords_Numpystyle

# Constants
NUM_MATCHES = 10
VISUALIZATION_SIZE = (20, 10)
POINT_SIZE = 10
LINE_WIDTH = 0.5
LINE_COLOR = 'r'
POINT_COLOR = 'b'

# Add project root to system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

def setup_model(checkpoint_path: str) -> PMatch:
    """
    Set up and configure the PMatch model.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Configured PMatch model
    """
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



def process_and_visualize_matches(
    model: PMatch,
    image1_path: str,
    image2_path: str,
    output_path: str
) -> None:
    """
    Process two images, find correspondences, and visualize matches.

    Args:
        model: PMatch model
        image1_path: Path to first image
        image2_path: Path to second image
        output_path: Path to save visualization
    """
    # Load images
    im1 = Image.open(image1_path)
    im2 = Image.open(image2_path)
    w1, h1 = im1.size
    w2, h2 = im2.size

    # Match features
    dense_matches, dense_certainty = model.match(im1, im2)
    sparse_matches, _ = model.sample(dense_matches, dense_certainty, NUM_MATCHES)

    def _process_keypoints(kpts, width, height):
        """Process keypoints from matches."""
        return np.stack((
            width * (kpts[:, 0] + 1) / 2,
            height * (kpts[:, 1] + 1) / 2
        ), axis=-1)

    a = 1
    sparse_matches = sparse_matches.transpose()
    sparse_matches = torch.from_numpy(sparse_matches).view([1, 4, NUM_MATCHES, 1])

    # Convert normalized coordinates to pixel coordinates
    matches_1 = unnorm_coords_Numpystyle(
        sparse_matches[:, :2],
        h=h1, w=w1
    ).squeeze().numpy().transpose()
    
    matches_2 = unnorm_coords_Numpystyle(
        sparse_matches[:, 2:],
        h=h2, w=w2
    ).squeeze().numpy().transpose()

    # Create and save first image visualization
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(im1)
    plt.scatter(matches_1[:, 0], matches_1[:, 1], c=POINT_COLOR, s=POINT_SIZE)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'assets/temp1.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Create and save second image visualization
    fig2 = plt.figure(figsize=(10, 10))
    plt.imshow(im2)
    plt.scatter(matches_2[:, 0], matches_2[:, 1], c=POINT_COLOR, s=POINT_SIZE)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'assets/temp2.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Read and combine images
    img1 = Image.open(os.path.join(project_root, 'assets/temp1.png'))
    img2 = Image.open(os.path.join(project_root, 'assets/temp2.png'))
    
    # Resize second image to match first image's height
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_w2 = int(w2 * (h1 / h2))
    img2 = img2.resize((new_w2, h1), Image.Resampling.LANCZOS)

    # Create combined image
    combined_w = w1 + new_w2
    combined_img = Image.new('RGB', (combined_w, h1))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (w1, 0))

    # Save final result
    combined_img.save(output_path, quality=95)

    # Clean up temporary files
    os.remove(os.path.join(project_root, 'assets/temp1.png'))
    os.remove(os.path.join(project_root, 'assets/temp2.png'))

    print(f"Visualization saved to {output_path}")

def main():
    """Main function to run the demo."""
    # Fixed paths
    checkpoint_path = os.path.join(project_root, "checkpoints/pmatch_mega.pth")
    image1_path = os.path.join(project_root, "assets/img1.jpg")
    image2_path = os.path.join(project_root, "assets/img2.jpg")
    output_path = os.path.join(project_root, "assets/demo.png")

    # Set up model
    model = setup_model(checkpoint_path)

    # Process images and create visualization
    with torch.no_grad():
        process_and_visualize_matches(
            model,
            image1_path,
            image2_path,
            output_path
        )

if __name__ == "__main__":
    main()
