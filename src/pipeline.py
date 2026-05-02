"""
pipeline.py
───────────
Full KYC Quality Detection Pipeline.

Runs all three quality checks on every image in the dataset:
  1. Blur    → Laplacian variance
  2. Glare   → HSV V-channel thresholding
  3. Framing → Canny + contour aspect ratio

Outputs:
  - Console progress with PASS/FAIL per image
  - outputs/quality_report.csv with full results

Usage:
  python src/pipeline.py
"""

import cv2
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config  import DATASET_ROOT, OUTPUT_CSV, OUTPUT_DIR, IMAGE_EXTENSIONS
from src.blur    import is_blurry,      get_blur_label
from src.glare   import has_glare,      get_glare_label
from src.framing import is_framing_ok,  get_framing_label
from src.parser  import parse_filename


# ─────────────────────────────────────────────────────────────────
# PROCESS A SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────

def process_image(img_path: Path) -> dict | None:
    """
    Run all quality checks on one image.

    Args:
        img_path: Path object pointing to the image file

    Returns:
        dict of results, or None if image could not be read
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run all 3 checks
    blur_score,   blurry    = is_blurry(gray)
    glare_ratio,  glare     = has_glare(img)
    aspect_ratio, framed_ok = is_framing_ok(gray)

    # PASS only if all 3 checks pass
    overall = "PASS" if (not blurry and not glare and framed_ok) else "FAIL"

    # Parse L/B/A/D from filename
    meta = parse_filename(img_path.name)

    return {
        # File info
        "filename":       img_path.name,
        "path":           str(img_path),
        # Dataset metadata
        "person_id":      meta["person_id"],
        "lighting":       meta["lighting"],
        "background":     meta["background"],
        "angle":          meta["angle"],
        "distance":       meta["distance"],
        # Blur
        "blur_score":     blur_score,
        "is_blurry":      blurry,
        "blur_label":     get_blur_label(blur_score),
        # Glare
        "glare_ratio":    glare_ratio,
        "has_glare":      glare,
        "glare_label":    get_glare_label(glare_ratio),
        # Framing
        "aspect_ratio":   aspect_ratio,
        "good_framing":   framed_ok,
        "framing_label":  get_framing_label(aspect_ratio),
        # Final verdict
        "overall":        overall,
    }


# ─────────────────────────────────────────────────────────────────
# PROCESS ALL IMAGES IN DATASET
# ─────────────────────────────────────────────────────────────────

def run_pipeline(dataset_root: str = DATASET_ROOT) -> list[dict]:
    """
    Run the full quality pipeline on all images in the dataset.

    Args:
        dataset_root: Path to the root dataset folder

    Returns:
        List of result dicts, one per image
    """
    all_images = [
        p for p in Path(dataset_root).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total = len(all_images)
    if total == 0:
        print(f"❌ No images found in '{dataset_root}'")
        return []

    print(f"\n🔍 Found {total} images in '{dataset_root}'")
    print("─" * 65)

    results = []
    failed_reads = 0

    for img_path in tqdm(all_images, desc="Processing", unit="img"):
        result = process_image(img_path)

        if result is None:
            failed_reads += 1
            tqdm.write(f"  ⚠️  Could not read: {img_path.name}")
            continue

        results.append(result)

        icon = "✅" if result["overall"] == "PASS" else "❌"
        tqdm.write(
            f"  {icon} {result['filename']:<45} | "
            f"Blur:{result['blur_score']:>8.1f} | "
            f"Glare:{result['glare_ratio']:.3f} | "
            f"Aspect:{result['aspect_ratio']:.2f} | "
            f"{result['overall']}"
        )

    if failed_reads:
        print(f"\n  ⚠️  {failed_reads} image(s) could not be read.")

    return results


# ─────────────────────────────────────────────────────────────────
# SAVE CSV REPORT
# ─────────────────────────────────────────────────────────────────

def save_report(results: list[dict], output_path: str = OUTPUT_CSV):
    """
    Save results to a CSV file.

    Args:
        results:     List of result dicts from run_pipeline()
        output_path: Where to save the CSV
    """
    if not results:
        print("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📄 Report saved → {output_path}")


# ─────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print a summary table of quality check results."""
    total     = len(results)
    if total == 0:
        return

    passed    = sum(1 for r in results if r["overall"] == "PASS")
    failed    = total - passed
    blurry    = sum(1 for r in results if r["is_blurry"])
    glare     = sum(1 for r in results if r["has_glare"])
    bad_frame = sum(1 for r in results if not r["good_framing"])

    print("\n" + "=" * 50)
    print("📊  QUALITY REPORT SUMMARY")
    print("=" * 50)
    print(f"  Total images   : {total}")
    print(f"  ✅ PASS        : {passed}  ({100*passed//total}%)")
    print(f"  ❌ FAIL        : {failed}  ({100*failed//total}%)")
    print("─" * 50)
    print(f"  🌀 Blurry      : {blurry}")
    print(f"  ☀️  Glare       : {glare}")
    print(f"  📐 Bad framing : {bad_frame}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Dataset folder '{DATASET_ROOT}' not found.")
        print("   Make sure you run this from your project root.")
        sys.exit(1)

    results = run_pipeline()
    print_summary(results)
    save_report(results)
    print("\n✅ Done! Open outputs/quality_report.csv to explore results.\n")
