from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parents[1]

OUT_DIR = BASE_DIR / "model_weights" / "m4_embedding" / "all-MiniLM-L6-v2"

OUT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=str(OUT_DIR),
    local_dir_use_symlinks=False,
)

print(f"M4 embedding model downloaded to: {OUT_DIR}")