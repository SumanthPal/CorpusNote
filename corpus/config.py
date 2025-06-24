from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# this is a document.db
DIAGRAMS_PATH = './diagrams'
DB_PATH = './research.db'
COLLECTION_NAME = 'documents'
IGNORE_DIRECTORIES = [
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
    # Add your ChromaDB path here if it's inside the project
    # e.g., "chroma_db" 
]

# Add file patterns to this list to exclude them.
# Uses standard glob matching (e.g., '*.log', 'temp_*')
IGNORE_FILES = [
    ".DS_Store",
    "*.log",
    "*.tmp",
    "*.swp",
    "*.swo",
    "thumbs.db"
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# MODEL SETTINGS
GEMINI_MODEL = 'gemini-1.5-flash'
GEMINI_IMG_MODEL = 'gemini-pro-vision'
MAX_RESULTS = 5
MAX_MEMORY = 10

MIN_CHUNK_LENGTH = 50
MAX_FILE_SIZE_MB = 100
CODE_EXTENSIONS = [
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    '.r', '.m', '.cs', '.sh', '.bash', '.zsh', '.fish',
    '.sql', '.jsx', '.tsx', '.vue', '.yaml', '.yml', '.json',
    '.xml', '.html', '.css', '.scss', '.less',
    '.dockerfile', '.makefile', '.cmake', '.gradle'
]
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.svg']
TEXT_EXTENSIONS = ['.txt', '.md', '.pdf']
SUPPORTED_EXTENSIONS = CODE_EXTENSIONS + IMAGE_EXTENSIONS + TEXT_EXTENSIONS


