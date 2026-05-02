"""
blur.py
───────
Blur Detection using Laplacian Variance.

How it works:
  1. Convert image to grayscale
  2. Apply Laplacian filter (edge/frequency detector)
  3. Compute variance of the result
  4. Low variance → few edges → blurry image

Research: PyImageSearch "Blur Detection with OpenCV" (2015)
Speed: ~5ms per image on CPU
"""

import cv2
import numpy as np
from src.config import BLUR_THRESHOLD


def compute_blur_score(gray: np.ndarray) -> float:
    """
    Compute the Laplacian variance of a grayscale image.

    Args:
        gray: Grayscale image as numpy array (H x W)

    Returns:
        float: Variance score. Higher = sharper. Lower = blurrier.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = laplacian.var()
    return round(score, 2)


def is_blurry(gray: np.ndarray, threshold: float = BLUR_THRESHOLD) -> tuple[float, bool]:
    """
    Check if a grayscale image is blurry.

    Args:
        gray:      Grayscale image
        threshold: Score below this = blurry (default from config)

    Returns:
        (score, is_blurry): score is the Laplacian variance,
                            is_blurry is True if image is blurry
    """
    score = compute_blur_score(gray)
    return score, score < threshold


def get_blur_label(score: float) -> str:
    """
    Convert a blur score into a human-readable severity label.

    Args:
        score: Laplacian variance score

    Returns:
        str: 'Sharp', 'Slightly Blurry', or 'Very Blurry'
    """
    if score >= BLUR_THRESHOLD:
        return "Sharp"
    elif score >= BLUR_THRESHOLD * 0.5:
        return "Slightly Blurry"
    else:
        return "Very Blurry"
