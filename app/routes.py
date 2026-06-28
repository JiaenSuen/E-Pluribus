from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

from .services.pipeline import run_ecology_pipeline
from .services.storage import save_uploaded_image
from .modules.registry import get_module_manifest

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@bp.get("/api/modules")
def modules():
    return jsonify({"modules": get_module_manifest()})


@bp.post("/api/analyze")
def analyze():
    query = (request.form.get("query") or "").strip()
    image = request.files.get("image")

    if not query:
        return jsonify({"error": "Query is required."}), 400
    if image is None or image.filename == "":
        return jsonify({"error": "One image file is required."}), 400

    try:
        saved = save_uploaded_image(image, current_app.config)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = run_ecology_pipeline(image_path=saved["path"], image_url=saved["url"], query=query)
    return jsonify(result)


@bp.get("/uploads/<path:filename>")
def uploaded_file(filename):
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    return send_from_directory(upload_dir, filename)
