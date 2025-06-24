#!/bin/bash
# restructure.sh - Convert to proper package structure

echo "Restructuring Corpus CLI as a package..."

# Create corpus package directory
mkdir -p corpus

# Move Python files into package
mv main.py corpus/__main__.py
mv chat.py corpus/chat.py
mv config_manager.py corpus/config_manager.py
mv diagram_generator.py corpus/diagram_generator.py
mv file_parser.py corpus/file_parser.py
mv web_parser.py corpus/web_parser.py

# Create __init__.py
cat > corpus/__init__.py << 'EOF'
"""Corpus CLI - Document indexing and chat with AI"""
__version__ = "1.1.0"
EOF

# Update imports in __main__.py
echo "Updating imports..."
sed -i.bak 's/from chat import/from .chat import/g' corpus/__main__.py
sed -i.bak 's/from file_parser import/from .file_parser import/g' corpus/__main__.py
sed -i.bak 's/from web_parser import/from .web_parser import/g' corpus/__main__.py
sed -i.bak 's/from diagram_generator import/from .diagram_generator import/g' corpus/__main__.py
sed -i.bak 's/from config import/from .config import/g' corpus/__main__.py

# Update imports in other files
for file in corpus/*.py; do
    if [ -f "$file" ] && [ "$file" != "corpus/__init__.py" ]; then
        sed -i.bak 's/from config import/from .config import/g' "$file"
        sed -i.bak 's/from file_parser import/from .file_parser import/g' "$file"
        sed -i.bak 's/from web_parser import/from .web_parser import/g' "$file"
        sed -i.bak 's/from chat import/from .chat import/g' "$file"
        sed -i.bak 's/from diagram_generator import/from .diagram_generator import/g' "$file"
    fi
done

# Clean up backup files
rm corpus/*.bak 2>/dev/null

# Move config if it exists
if [ -f "config.py" ]; then
    mv config.py corpus/config.py
fi
if [ -f "config.example.py" ]; then
    cp config.example.py corpus/config.example.py
fi

# Create new setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="corpus-cli",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "chromadb>=0.4.0",
        "google-generativeai>=0.3.0",
        "pypdf>=3.0.0",
        "python-docx>=1.0.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "githubkit>=0.10.0",
        "psutil>=5.9.0",
        "watchdog>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "corpus=corpus.__main__:app",
        ],
    },
    python_requires=">=3.8",
    package_data={
        "corpus": ["config.example.py"],
    },
    include_package_data=True,
)
