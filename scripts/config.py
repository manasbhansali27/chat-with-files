"""
config.py
---------
Central configuration for paths, extensions, scanning rules, and model settings.
All modules (scanner, embedder, output) import from here.
"""

import os

# =====================
# DATABASE & INDICES
# =====================
SQLITE_PATH = os.path.join("indices", "sqlite.db")

FAISS_TEXT_INDEX = os.path.join("indices", "faiss_text.index")
FAISS_IMAGE_INDEX = os.path.join("indices", "faiss_image.index")

# =====================
# SUPPORTED EXTENSIONS
# =====================
TEXT_EXTS = [
    ".pdf", ".docx", ".pptx", ".txt", ".py", ".ipynb",
    ".cpp", ".c", ".h", ".java", ".js", ".ts",
    ".html", ".css", ".json", ".md", ".csv"
]

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

# =====================
# FOLDER SCANNING
# =====================
FOLDERS_TO_SCAN = [
    "C:/Users/mmbha/Desktop",
    "C:/Users/mmbha/Downloads",
    "C:/Users/mmbha/Music",
    "C:/Users/mmbha/Pictures",
    "C:/Users/mmbha/Videos",
]

EXCLUDE_FOLDERS = [
    "C:/Users/mmbha\\AppData",
    "c:/Users/mmbha\\.ipython",
    "C:/Users/mmbha\\.cache",
    "C:/Users/mmbha\\.conda",
    "C:/Users/mmbha\\.anaconda3",
    "C:/Users/mmbha\\.vscode",
    "C:/Users/mmbha\\.rustup",
    "C:/Users/mmbha\\.cargo",
    "C:/Users/mmbha\\.git",
    "C:/Users/mmbha\\.hg",
    "C:/Users/mmbha\\ venv",
    "C:/Users/mmbha\\.venv",
    "C:/Users/mmbha\\.ipynb_checkpoints",
    "C:/Users/mmbha\\.arduinoIDE",
    "C:/Users/mmbha\\.jupyter",
    "C:/Users/mmbha\\.continuum",
    "c:/Users/mmbha\\.keras",
    "C:/Users/mmbha\\.matplotlib",
    "c:/Users/mmbha\\.mchp_packs",
    "c:/Users/mmbha\\.mplabcomm",
    "C:/Users/mmbha\\.ollama",
    "C:/Users/mmbha\\.streamlit",
    "C:/Users/mmbha\\ ansel"
]

SKIP_DIR_NAMES = {
    "venv", ".venv", "env", ".env",
    "__pycache__", ".git", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"
}

# =====================
# MODEL SETTINGS
# =====================
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"                # SentenceTransformer model
IMAGE_EMBED_MODEL = "google/siglip-base-patch16-224" # SigLIP model
OLLAMA_EMBED_MODEL = "mistral:7b"                    # Ollama fallback
EASYOCR_LANGS = ["en"]                               # OCR languages
OCR_CONFIDENCE = 0.5                                 # Min confidence threshold

