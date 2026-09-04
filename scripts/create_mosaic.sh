#!/bin/bash
set -euo pipefail

n_columns=$1
n_rows=$2
total_width=$3
source_image_folder=$4

shopt -s nullglob nocaseglob
image_files=("$source_image_folder"/*.jpg "$source_image_folder"/*.jpeg "$source_image_folder"/*.png)
shopt -u nocaseglob

total_images=${#image_files[@]}
required_images=$((n_columns * n_rows))
echo "Total images: $total_images / required: $required_images"

if [ "$total_images" -lt "$required_images" ]; then
  echo "Not enough images to create the specified mosaic dimensions."
  exit 1
fi

# Compute per-tile width from the desired total output width
tile_width=$(( total_width / n_columns ))
echo "Total output width: ${total_width}px -> tile width: ${tile_width}px (${n_columns} columns)"

# Guard against ImageMagick's default 32000px dimension limit
max_dimension=32000
if [ "$total_width" -gt "$max_dimension" ]; then
  echo "Error: requested total width ($total_width) exceeds ImageMagick's ${max_dimension}px limit."
  exit 1
fi

temp_dir=$(mktemp -d)
echo "Resizing images"

# Space-safe random selection
mapfile -t selected_images < <(printf '%s\n' "${image_files[@]}" | shuf -n "$required_images")

echo "Selected images:"
printf '  %s\n' "${selected_images[@]}"

# --- progress bar helper ---
draw_progress() {
  local current=$1 total=$2 width=40
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  local bar
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar+=$(printf '%*s' "$empty" '' | tr ' ' '-')
  local percent=$(( current * 100 / total ))
  printf '\r[%s] %3d%% (%d/%d)' "$bar" "$percent" "$current" "$total"
}
# ---------------------------

i=0
count=0
for image in "${selected_images[@]}"; do
  ext="${image##*.}"
  convert "$image" -resize "${tile_width}x" "$temp_dir/tile_$i.$ext"
  i=$((i + 1))
  count=$((count + 1))
  draw_progress "$count" "$required_images"
done
echo ""

echo "Creating mosaic"
montage "$temp_dir"/tile_* -tile "${n_columns}x${n_rows}" -geometry +0+0 output.jpg

rm -rf "$temp_dir"
echo "Mosaic created successfully."
