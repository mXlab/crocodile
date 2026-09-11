#!/usr/bin/env python3
"""Ensures each grid_cell widget in live_pipeline/crocodile-control-panel.json
uses the machine-independent theme-class scheme for its thumbnail image,
instead of a literal background-image:url(<path>) baked into its own css.

Why: Open Stage Control's css property does not support JS{}/#{}
templating (confirmed empirically -- see the open-stage-control skill's
widgets.md), so a per-cell image can't be computed at load time from
inside the session JSON. The old approach baked a literal, machine-specific
absolute path into every one of the 100 grid_cell widgets, which broke on
every new machine/checkout and needed its own resync step.

Instead: emotion_grid/build_grid.py generates emotion_grid/data/theme.css,
one ".grid-cell-<id> { background-image: url(...) }" rule per image, using
a path relative to theme.css's own location (thumbnails/ lives right next
to it) -- confirmed against the actual server source
(src/server/node/server.mjs's resolvePath) that a theme's relative url()s
resolve against the theme file's own directory, independent of the
session's location or any per-client state. run_control_panel.sh loads it
via --theme. Each widget just needs a stable `class: grid-cell-<id>;` line
in its css (a real, literal OSC syntax -- see widget.mjs's "extra css
class property" handling) to pick up that rule -- this class name is
derived purely from the id and never changes across machines, so unlike
the old approach this script should rarely need re-running (only if
emotion_grid/build_grid.py's cell ids themselves change).

Usage:
    python emotion_grid/update_panel_thumbnails.py
"""

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches the old inline-url css this script used to write, so it can be
# stripped even from panels generated before the theme.css scheme existed.
OLD_INLINE_IMAGE_CSS = re.compile(r'background-image:url\([^)]*\);background-size:cover;background-position:center;')


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
        description="Ensure the control panel's grid cells reference theme.css classes instead of literal image paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--panel', default=str(REPO_ROOT / 'live_pipeline' / 'crocodile-control-panel.json'))
    args = parser.parse_args()

    with open(args.panel) as f:
        panel = json.load(f)

    n_updated = 0
    n_already = 0
    for widget in iter_widgets(panel['content']):
        widget_id = widget.get('id', '')
        if not widget_id.startswith('grid_cell_'):
            continue
        cell_id = widget_id[len('grid_cell_'):]
        css = widget.get('css', '')

        class_lines = f"class: grid-cell;\nclass: grid-cell-{cell_id};\n"
        if css.startswith(class_lines):
            n_already += 1
            continue

        # Strip a pre-existing class-lines prefix (in case cell_id or the
        # class scheme itself changed) and the old inline-url css, whichever
        # is present, leaving only the dynamic per-instance rules (the
        # selection-highlight border/box-shadow, driven by @{selected_id}).
        remainder = re.sub(r'^(?:class:[^\n]*\n)+', '', css)
        remainder = OLD_INLINE_IMAGE_CSS.sub('', remainder)

        widget['css'] = class_lines + remainder
        n_updated += 1

    # ensure_ascii=True matches the panel file's own current escaping convention (confirmed via
    # round-trip byte comparison) -- keeps this script's diff to just the intended css changes
    # instead of also reflowing every non-ASCII character elsewhere in the file.
    with open(args.panel, 'w') as f:
        json.dump(panel, f, indent=2, ensure_ascii=True)

    print(f"Updated {n_updated} grid cell(s), {n_already} already using the theme-class scheme, in {args.panel}")


if __name__ == '__main__':
    main()
