#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

echo "Downloading PMatch pre-trained models..."

# URLs for the models
INDOOR_MODEL_URL="https://huggingface.co/datasets/shngjz/ce29d0e9486d476eb73163644b050222/resolve/main/checkpoints/pmatch_scannet.pth"
OUTDOOR_MODEL_URL="https://huggingface.co/datasets/shngjz/ce29d0e9486d476eb73163644b050222/resolve/main/checkpoints/pmatch_mega.pth"

# Function to download with progress using curl
download_file() {
    local url=$1
    local output=$2
    echo "Downloading $(basename $output)..."
    if curl -L --progress-bar "$url" -o "$output"; then
        echo "✓ Successfully downloaded $(basename $output)"
    else
        echo "✗ Failed to download $(basename $output)"
        return 1
    fi
}

# Download indoor model
download_file "$INDOOR_MODEL_URL" "checkpoints/pmatch_scannet.pth"

# Download outdoor model
download_file "$OUTDOOR_MODEL_URL" "checkpoints/pmatch_mega.pth"

echo
echo "Download complete! Models are saved in the 'checkpoints' directory."
echo "- Indoor model (ScanNet): checkpoints/pmatch_scannet.pth"
echo "- Outdoor model (MegaDepth): checkpoints/pmatch_mega.pth"
