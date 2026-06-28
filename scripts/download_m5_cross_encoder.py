from pathlib import Path
from sentence_transformers import CrossEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "model_weights" / "m5_reranker" / "ms-marco-MiniLM-L-6-v2"

OUT_DIR.mkdir(parents=True, exist_ok=True)

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model.save(str(OUT_DIR))

print(f"Saved M5 CrossEncoder to: {OUT_DIR}")
