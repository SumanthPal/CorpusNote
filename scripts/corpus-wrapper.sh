#!/bin/bash
# Global wrapper for corpus command
# Save this as 'corpus' and add to your PATH

# Find the corpus installation directory
CORPUS_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

# Check if virtual environment exists
if [ ! -d "$CORPUS_DIR/env" ]; then
    echo "Error: Virtual environment not found at $CORPUS_DIR/env"
    echo "Please run the installation script first."
    exit 1
fi

# Run corpus with the virtual environment
"$CORPUS_DIR/env/bin/python" "$CORPUS_DIR/env/bin/corpus" "$@"