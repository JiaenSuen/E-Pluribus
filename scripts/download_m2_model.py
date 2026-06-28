from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parents[1]

snapshot_download(
    repo_id="nlpconnect/vit-gpt2-image-captioning",
    local_dir=BASE_DIR / "model_weights" / "m2_vit_gpt2_image_captioning",
    local_dir_use_symlinks=False,
)

print("M2 model downloaded.")