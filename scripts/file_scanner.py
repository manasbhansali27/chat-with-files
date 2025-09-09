"""
file_scanner.py

Scans configured directories for text, image, and video files,
stores metadata in a SQLite database, and avoids excluded/virtualenv paths.

This module is part of the lightweight local AI assistant project.
"""

import os
import sqlite3
import time
import logging
from pathlib import Path
from tqdm import tqdm
from config import (
    SQLITE_PATH,
    FOLDERS_TO_SCAN,
    EXCLUDE_FOLDERS,
    SKIP_DIR_NAMES,
    TEXT_EXTS,
    IMAGE_EXTS,
    VIDEO_EXTS,
)

# =====================
# LOGGING SETUP
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =====================
# DB INIT
# =====================
def init_db():
    """
    Initialize SQLite DB with a `files` table.
    Creates DB directory if it does not exist.
    """
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            type TEXT,
            size INTEGER,
            modified REAL
        )
    """)
    conn.commit()
    return conn

# =====================
# HELPERS
# =====================
def get_file_type(file_path: str) -> str | None:
    """
    Return the type of file based on its extension.

    Args:
        file_path (str): Path to file.

    Returns:
        str | None: "text", "image", "video", or None if unsupported.
    """
    ext = Path(file_path).suffix.lower()
    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None

def looks_like_virtualenv_root(path: Path) -> bool:
    """Detect if a folder looks like a Python virtual environment root."""
    try:
        if (path / "pyvenv.cfg").exists():
            return True
        if (path / "Lib" / "site-packages").is_dir():
            return True
        if (path / "Scripts" / "activate").exists():
            return True
        if (path / "bin" / "activate").exists():
            return True
    except Exception:
        return False
    return False

def should_skip_folder(folder_path: str) -> bool:
    """Return True if a folder should be skipped based on rules."""
    p = Path(folder_path)

    # Skip by folder name
    if any(part.lower() in SKIP_DIR_NAMES for part in p.parts):
        return True

    # Skip if it looks like a venv root
    if looks_like_virtualenv_root(p):
        return True

    # Skip explicit exclude folders
    if any(folder_path.startswith(excl) for excl in EXCLUDE_FOLDERS):
        return True

    return False

# =====================
# MAIN SCANNER
# =====================
def scan_folders():
    """
    Scan configured folders, insert file metadata into SQLite DB.
    """
    conn = init_db()
    cursor = conn.cursor()

    counts = {"text": 0, "image": 0, "video": 0}
    ext_counts = {}
    total = 0

    # Collect candidate files
    file_list = []
    for folder in FOLDERS_TO_SCAN:
        folder = os.path.expanduser(folder)
        if not os.path.exists(folder):
            logger.warning("Folder not found: %s", folder)
            continue

        for root, dirs, files in os.walk(folder, topdown=True):
            # prune dirs
            dirs[:] = [d for d in dirs if not should_skip_folder(os.path.join(root, d))]
            for f in files:
                file_path = os.path.join(root, f)
                if get_file_type(file_path):
                    file_list.append(file_path)

    # Process files with progress bar
    for file_path in tqdm(file_list, desc="Scanning files", unit="file"):
        try:
            ftype = get_file_type(file_path)
            if not ftype:
                continue

            size = os.path.getsize(file_path)
            modified = os.path.getmtime(file_path)

            cursor.execute(
                """
                INSERT OR REPLACE INTO files (path, type, size, modified)
                VALUES (?, ?, ?, ?)
                """,
                (file_path, ftype, size, modified),
            )

            counts[ftype] += 1
            ext = Path(file_path).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total += 1

        except Exception as e:
            logger.error("Skipping %s: %s", file_path, e)

    conn.commit()
    conn.close()

    # Summary
    logger.info("Indexed %d files into %s", total, SQLITE_PATH)
    for t, c in counts.items():
        logger.info("  %s files: %d", t.capitalize(), c)

    logger.info("Extension breakdown:")
    for ext, c in sorted(ext_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", ext, c)

# =====================
# RUN
# =====================
if __name__ == "__main__":
    start = time.time()
    scan_folders()
    logger.info("Scan completed in %.2fs", time.time() - start)
