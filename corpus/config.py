from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Import the ConfigManager to get values from ~/.corpus/config.json
try:
    from .config_manager import get_config
    _config_manager = get_config()
    _use_config_manager = True
except ImportError:
    # Fallback if config_manager isn't available
    _config_manager = None
    _use_config_manager = False

def _get_config_value(key: str, default_value):
    """Get value from ConfigManager if available, otherwise use default"""
    if _use_config_manager and _config_manager:
        return _config_manager.get(key, default_value)
    return os.getenv(key, default_value)

# API Keys - Always check environment first, then config manager
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or _get_config_value("GEMINI_API_KEY", "")

# Storage paths
DB_PATH = _get_config_value("DB_PATH", './research.db')
DIAGRAMS_PATH = _get_config_value("DIAGRAMS_PATH", './diagrams')
COLLECTION_NAME = _get_config_value("COLLECTION_NAME", 'documents')

# Text processing configuration
CHUNK_SIZE = _get_config_value("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _get_config_value("CHUNK_OVERLAP", 200)
MIN_CHUNK_LENGTH = _get_config_value("MIN_CHUNK_LENGTH", 50)
MAX_FILE_SIZE_MB = _get_config_value("MAX_FILE_SIZE_MB", 100)

# Model settings
GEMINI_MODEL = _get_config_value("GEMINI_MODEL", 'gemini-1.5-flash')
GEMINI_IMG_MODEL = _get_config_value("GEMINI_IMG_MODEL", 'gemini-pro-vision')

# Search and memory settings
MAX_RESULTS = _get_config_value("MAX_RESULTS", 5)
MAX_MEMORY = _get_config_value("MAX_MEMORY", 10)

# File extensions
CODE_EXTENSIONS = _get_config_value("CODE_EXTENSIONS", [
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    '.r', '.m', '.cs', '.sh', '.bash', '.zsh', '.fish',
    '.sql', '.jsx', '.tsx', '.vue', '.yaml', '.yml', '.json',
    '.xml', '.html', '.css', '.scss', '.less',
    '.dockerfile', '.makefile', '.cmake', '.gradle'
])

IMAGE_EXTENSIONS = _get_config_value("IMAGE_EXTENSIONS", [
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.svg'
])

TEXT_EXTENSIONS = _get_config_value("TEXT_EXTENSIONS", [
    '.txt', '.md', '.pdf', '.docx', '.rtf'
])

# Directory and file ignore patterns
IGNORE_DIRECTORIES = _get_config_value("IGNORE_DIRECTORIES", [
    "__pycache__",
    "node_modules",
    ".git", 
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "env",
    "dist", 
    "build",
    "eggs",
    ".eggs",
])

IGNORE_FILES = _get_config_value("IGNORE_FILES", [
    ".DS_Store",
    "*.log",
    "*.tmp", 
    "*.swp",
    "*.swo",
    "thumbs.db"
])

# Computed values
SUPPORTED_EXTENSIONS = CODE_EXTENSIONS + IMAGE_EXTENSIONS + TEXT_EXTENSIONS

# Helper function to update config values
def update_config(key: str, value):
    """Update a configuration value and save to config manager"""
    if _use_config_manager and _config_manager:
        _config_manager.set(key, value)
        # Also update the module-level variable
        globals()[key] = value
        return True
    return False