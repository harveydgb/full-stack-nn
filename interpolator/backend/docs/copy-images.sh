#!/bin/bash
# Copy performance profiling graphs to Sphinx static files directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="$SCRIPT_DIR/../outputs"
STATIC_DIR="$SCRIPT_DIR/source/_static/images"

# Create images directory if it doesn't exist
mkdir -p "$STATIC_DIR"

# Copy PNG files
if [ -d "$OUTPUTS_DIR" ]; then
    cp "$OUTPUTS_DIR"/*.png "$STATIC_DIR/" 2>/dev/null
    echo "Copied images from $OUTPUTS_DIR to $STATIC_DIR"
    ls -lh "$STATIC_DIR"/*.png 2>/dev/null || echo "No PNG files found to copy"
else
    echo "Error: Outputs directory not found at $OUTPUTS_DIR"
    exit 1
fi

