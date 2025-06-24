#!/bin/bash
# Script to add corpus alias to shell configuration

# Get the current directory (corpus installation path)
CORPUS_DIR="$(pwd)"

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    echo "Unsupported shell. Please add the alias manually."
    exit 1
fi

# Create the corpus function
CORPUS_FUNCTION="
# Corpus CLI
corpus() {
    cd \"$CORPUS_DIR\" && source env/bin/activate && command corpus \"\$@\"
    cd - > /dev/null  # Return to previous directory
}"

# Check if corpus function already exists
if grep -q "corpus()" "$SHELL_CONFIG"; then
    echo "⚠️  Corpus function already exists in $SHELL_CONFIG"
    echo "   Remove the old one if you want to update the path."
else
    # Add to shell config
    echo "" >> "$SHELL_CONFIG"
    echo "$CORPUS_FUNCTION" >> "$SHELL_CONFIG"
    echo "✅ Added corpus function to $SHELL_CONFIG"
fi

echo ""
echo "To use corpus immediately, run:"
echo "  source $SHELL_CONFIG"
echo ""
echo "Or just open a new terminal window."
echo ""
echo "Then you can use 'corpus' from anywhere!"