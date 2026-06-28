from pathlib import Path

REQUIRED = [
    "run.py",
    "requirements.txt",
    "app/routes.py",
    "app/services/pipeline.py",
    "app/modules/registry.py",
    "app/templates/index.html",
    "app/static/css/styles.css",
    "app/static/js/app.js",
]

root = Path(__file__).resolve().parents[1]
missing = [path for path in REQUIRED if not (root / path).exists()]

if missing:
    raise SystemExit(f"Missing files: {missing}")

print("Structure OK")
