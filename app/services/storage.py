from pathlib import Path
from uuid import uuid4

from flask import url_for
from werkzeug.utils import secure_filename


def _allowed(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def save_uploaded_image(file_storage, config):
    original = secure_filename(file_storage.filename or "observation.jpg")
    if not _allowed(original, config["ALLOWED_EXTENSIONS"]):
        allowed = ", ".join(sorted(config["ALLOWED_EXTENSIONS"]))
        raise ValueError(f"Unsupported image type. Allowed: {allowed}")

    upload_dir = Path(config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(original).suffix.lower()
    filename = f"{uuid4().hex}{suffix}"
    path = upload_dir / filename
    file_storage.save(path)

    return {
        "path": str(path),
        "filename": filename,
        "original_filename": original,
        "url": url_for("main.uploaded_file", filename=filename),
    }
