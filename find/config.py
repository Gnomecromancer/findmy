"""Configuration for find — paths, model, indexing defaults."""
import os
from pathlib import Path

# Index storage
FIND_DIR = Path(os.environ.get("FIND_DIR", Path.home() / ".find"))
INDEX_PATH = FIND_DIR / "index.faiss"
META_PATH = FIND_DIR / "meta.db"
MODEL_CACHE = FIND_DIR / "models"

# Embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 64           # chunks per GPU batch
MAX_CHUNK_TOKENS = 512    # hard cap per chunk
CHUNK_OVERLAP = 64        # token overlap between chunks

# File types to index
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".java", ".kt", ".swift", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".bat", ".ps1",
    ".md", ".txt", ".rst", ".org",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".conf",
    ".html", ".css", ".scss", ".sql",
    ".csv", ".tsv",
}

# Directories to always skip
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", "dist", "build", ".eggs", "*.egg-info",
    ".idea", ".vscode", ".DS_Store",
    "target",           # Rust/Java build
    ".next", ".nuxt",   # JS frameworks
    "coverage",
}

MAX_FILE_SIZE_MB = 2.0   # skip files larger than this
