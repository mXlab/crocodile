"""
Test script for build_grid.py's grid_layout.json export
"""
import json
import os
import tempfile

import pandas as pd

from build_grid import build_grid_layout

print("="*80)
print("BUILD_GRID_LAYOUT TEST")
print("="*80)

# Synthetic manifest: 2 emotions, uneven counts (2 vs 1) to exercise the
# rectangular-padding path.
manifest_df = pd.DataFrame([
    {"id": "ang_00", "emotion": "ang", "thumbnail_path": "thumbnails/ang/ang_00.png"},
    {"id": "ang_01", "emotion": "ang", "thumbnail_path": "thumbnails/ang/ang_01.png"},
    {"id": "joy_00", "emotion": "joy", "thumbnail_path": "thumbnails/joy/joy_00.png"},
])
labels_df = pd.DataFrame([
    {"code": "ang", "full_name": "anger"},
    {"code": "joy", "full_name": "joy"},
])

with tempfile.TemporaryDirectory() as output_dir:
    layout_path = build_grid_layout(manifest_df, labels_df, output_dir)
    assert os.path.isfile(layout_path)

    with open(layout_path) as f:
        layout = json.load(f)

    assert layout["emotions"] == [
        {"code": "ang", "name": "anger"},
        {"code": "joy", "name": "joy"},
    ], layout["emotions"]
    assert layout["images_per_emotion"] == 2, layout["images_per_emotion"]
    assert len(layout["cells"]) == 4, len(layout["cells"])  # 2 emotions x 2 rows

    # row 0: ang_00, joy_00
    assert layout["cells"][0]["id"] == "ang_00"
    assert layout["cells"][0]["thumbnailPath"] == os.path.join(output_dir, "thumbnails/ang/ang_00.png")
    assert layout["cells"][1]["id"] == "joy_00"
    # row 1: ang_01, then padding (joy only has 1 image)
    assert layout["cells"][2]["id"] == "ang_01"
    assert layout["cells"][3] is None

    print("\n✓ emotions, images_per_emotion, row-major ordering, and padding are all correct")

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
