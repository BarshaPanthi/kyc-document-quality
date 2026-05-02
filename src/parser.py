"""
parser.py
─────────
Filename Metadata Parser for the UK Passport Dataset.

Filename format:
    P_UK_{person_id}_L{lighting}_B{background}_A{angle}_D{distance}.jpg

Example:
    P_UK_427583910_L1_B2_A3_D1.jpg
    → person_id  = "427583910"
    → lighting   = "L1"   (condition 1 of 4)
    → background = "B2"   (condition 2 of 4)
    → angle      = "A3"   (condition 3 of 3)
    → distance   = "D1"   (condition 1 of 2)

Variable meanings:
    L1–L4 : Lighting conditions (bright, dim, overhead, natural)
    B1–B4 : Background types (white, dark, textured, outdoor)
    A1–A3 : Camera angles (straight, slight tilt, steep tilt)
    D1–D2 : Distance from camera (close, far)
"""

import re


# Regex pattern matching the filename convention
_FILENAME_PATTERN = re.compile(
    r"P_UK_(?P<person_id>\d+)_L(?P<L>\d+)_B(?P<B>\d+)_A(?P<A>\d+)_D(?P<D>\d+)"
)


def parse_filename(filename: str) -> dict:
    """
    Extract metadata from a dataset filename.

    Args:
        filename: Image filename (with or without path/extension)
                  e.g. "P_UK_427583910_L1_B1_A1_D1.jpg"

    Returns:
        dict with keys: person_id, lighting, background, angle, distance
        Returns '?' values if filename doesn't match the pattern.
    """
    match = _FILENAME_PATTERN.search(filename)

    if not match:
        return {
            "person_id":  "unknown",
            "lighting":   "?",
            "background": "?",
            "angle":      "?",
            "distance":   "?",
        }

    return {
        "person_id":  match.group("person_id"),
        "lighting":   f"L{match.group('L')}",
        "background": f"B{match.group('B')}",
        "angle":      f"A{match.group('A')}",
        "distance":   f"D{match.group('D')}",
    }


def parse_batch(filenames: list[str]) -> list[dict]:
    """
    Parse metadata from a list of filenames.

    Args:
        filenames: List of image filenames

    Returns:
        List of metadata dicts (one per filename)
    """
    return [parse_filename(f) for f in filenames]
