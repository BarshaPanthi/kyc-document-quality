"""
config.py
─────────
Central configuration for the KYC Quality Checker.
All thresholds and paths are defined here.
Change values here to tune detection sensitivity.
"""

import os

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────

# Root folder of the dataset (relative to project root)
DATASET_ROOT = r"C:\kyc-document-quality\uk_password"

# Where to save the output CSV report
OUTPUT_DIR   = "outputs"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "quality_report.csv")

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────
# BLUR DETECTION — Laplacian Variance
# ─────────────────────────────────────────────────────────────────

# Images with Laplacian variance BELOW this are considered blurry.
# Higher = stricter (more images flagged as blurry).
# Typical range: 50 (lenient) → 200 (strict)
BLUR_THRESHOLD = 100.0


# ─────────────────────────────────────────────────────────────────
# GLARE DETECTION — HSV V-channel Thresholding
# ─────────────────────────────────────────────────────────────────

# Pixels with HSV Value > this are considered "overexposed / glare"
# Range: 0–255. 240 catches only extreme highlights.
GLARE_V_VALUE = 240

# If more than this fraction of pixels are glare pixels → reject
# 0.10 = 10% of image. Lower = stricter.
GLARE_THRESHOLD = 0.10


# ─────────────────────────────────────────────────────────────────
# FRAMING DETECTION — Canny + Contour Aspect Ratio
# ─────────────────────────────────────────────────────────────────

# Canny edge detection thresholds
CANNY_LOW  = 50
CANNY_HIGH = 150

# GaussianBlur kernel size before Canny
GAUSSIAN_KERNEL = (5, 5)

# Valid passport/ID aspect ratio range (width / height)
# UK passport landscape: ~1.42  |  ID card: ~1.58
ASPECT_MIN = 1.3
ASPECT_MAX = 2.0
