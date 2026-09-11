#!/usr/bin/env python3
"""Syncs each grid_cell widget's thumbnail image path in
live_pipeline/crocodile-control-panel.json to match the current machine's
actual absolute paths, read from emotion_grid/data/grid_layout.json.

Why this is needed: the control panel references thumbnails via CSS
background-image, and Open Stage Control's css property does not support
JS{}/#{} templating (confirmed empirically -- see the open-stage-control
skill's widgets.md). So each grid cell's image path is baked into the
session JSON as a literal string rather than computed at load time, the
same way grid_layout.json itself bakes in absolute paths. Re-run this
script any time build_grid.py regenerates the grid (paths may shift), or
when opening this session on a different machine/checkout where the
absolute paths baked in from a previous machine won't resolve.

Deliberately does NOT bake in a http://host:port prefix -- a path
starting with / is resolved by the browser against whatever origin
actually served the page, so the server's port never needs to be
hardcoded or kept in sync with this script. Only the machine-specific
filesystem path needs syncing.

Usage:
    python emotion_grid/update_panel_thumbnails.py
"""

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def iter_widgets(node):
    """Yields every widget dict in the tree, recursing into nested 'widgets' and 'tabs' arrays
    (a tab/panel container's nested tab-panels live under 'tabs', separate from 'widgets')."""
    if isinstance(node, dict):
        yield node
        for child in node.get('widgets', []):
            yield from iter_widgets(child)
        for child in node.get('tabs', []):
            yield from iter_widgets(child)
    elif isinstance(node, list):
        for item in node:
            yield from iter_widgets(item)


def main():
    parser = argparse.ArgumentParser(
        description="Sync the control panel's grid cell thumbnail paths to the current grid_layout.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grid-layout', default=str(REPO_ROOT / 'emotion_grid' / 'data' / 'grid_layout.json'))
    parser.add_argument('--panel', default=str(REPO_ROOT / 'live_pipeline' / 'crocodile-control-panel.json'))
    args = parser.parse_args()

    with open(args.grid_layout) as f:
        layout = json.load(f)
    thumbnail_by_id = {cell['id']: cell['thumbnailPath'] for cell in layout['cells'] if cell}
    print(f"Loaded {len(thumbnail_by_id)} thumbnail paths from {args.grid_layout}")

    with open(args.panel) as f:
        panel = json.load(f)

    n_updated = 0
    n_missing = 0
    seen_ids = set()
    for widget in iter_widgets(panel['content']):
        widget_id = widget.get('id', '')
        if not widget_id.startswith('grid_cell_'):
            continue
        cell_id = widget_id[len('grid_cell_'):]
        seen_ids.add(cell_id)
        if cell_id not in thumbnail_by_id:
            print(f"  WARNING: {widget_id} has no matching cell in grid_layout.json -- leaving as-is")
            n_missing += 1
            continue
        css = widget.get('css', '')
        new_css, n = re.subn(
            r'background-image:url\([^)]*\)',
            f'background-image:url({thumbnail_by_id[cell_id]})',
            css)
        if n == 0:
            print(f"  WARNING: {widget_id}'s css has no background-image:url(...) to replace -- leaving as-is")
            continue
        widget['css'] = new_css
        n_updated += 1

    n_unmatched_layout_ids = len(set(thumbnail_by_id) - seen_ids)
    if n_unmatched_layout_ids:
        print(f"  Note: {n_unmatched_layout_ids} grid_layout.json cell(s) have no matching "
              f"grid_cell_* widget in the panel (added images since the panel was last built?)")

    # ensure_ascii=True matches the panel file's own current escaping convention (confirmed via
    # round-trip byte comparison) -- keeps this script's diff to just the intended css changes
    # instead of also reflowing every non-ASCII character elsewhere in the file.
    with open(args.panel, 'w') as f:
        json.dump(panel, f, indent=2, ensure_ascii=True)

    print(f"Updated {n_updated} grid cell thumbnail paths in {args.panel} ({n_missing} missing)")


if __name__ == '__main__':
    main()
