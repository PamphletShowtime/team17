#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p images
mkdir -p results
RESULT_DIR="results/result_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "Capturing 3 frames at 5fps (approximately 0.2s interval per frame)..."

gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=3 ! \
    videoconvert ! \
    videorate ! "video/x-raw,framerate=5/1" ! \
    jpegenc ! \
    multifilesink location="images/frame-%05d.jpg"

echo "Capture completed. Starting pose estimation..."

export PYTHONPATH=$(pwd)

for img in images/frame-*.jpg; do
    echo "Processing: $img"
    python3 scripts/test.py "$img" "$RESULT_DIR"
done

echo "All 3 images processed. Pose estimation completed."
echo "Result images and numpy files are in $RESULT_DIR"

echo "Extracting features..."
mkdir -p scripts/features
python3 scripts/extract_features.py "$RESULT_DIR" "scripts/features"
echo "Features extracted and saved in the 'scripts/features' directory."

exit 0

