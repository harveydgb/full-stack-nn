#!/bin/bash
# Build Sphinx documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/../docs"
SOURCE_DIR="$DOCS_DIR/source"
BUILD_DIR="$DOCS_DIR/build"

# Check if Sphinx is installed (sphinx-build will fail with clear error if not)
if ! command -v sphinx-build &> /dev/null 2>&1; then
    echo "Error: Sphinx is not installed. Install with: pip install -e '.[docs]'"
    exit 1
fi

# Navigate to docs directory
cd "$DOCS_DIR"

# Clean previous builds (optional - comment out to keep previous builds)
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Build documentation
echo "Building documentation..."
echo "Source: $SOURCE_DIR"
echo "Output: $BUILD_DIR/html"
echo ""

sphinx-build -b html "$SOURCE_DIR" "$BUILD_DIR/html"

# Display results
INDEX_PATH="$BUILD_DIR/html/index.html"
    echo ""
    echo "=========================================="
    echo "Documentation built successfully!"
    echo "=========================================="
    echo ""
    echo "Output location: $INDEX_PATH"
    echo ""
    echo "To view in browser:"
    echo "  file://$INDEX_PATH"
    echo ""
echo "Or run:"
echo "  open $INDEX_PATH  # macOS"
echo "  xdg-open $INDEX_PATH  # Linux"
    
# Try to open in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "Open in browser now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$INDEX_PATH"
    fi
fi
