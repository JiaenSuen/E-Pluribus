import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from app.modules.base import ensure_context

MODULE_ID = "m2_context"
MODULE_NAME = "M2 ViT-GPT2 Visual Context"
MODULE_ORDER = 20
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "Image captioning context module. Does not consume M1 outputs."

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = Path(
    os.getenv(
        "M2_MODEL_PATH",
        BASE_DIR / "model_weights" / "m2_vit_gpt2_image_captioning",
    )
)

DEVICE = os.getenv("M2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.getenv("M2_MAX_LENGTH", "24"))
NUM_BEAMS = int(os.getenv("M2_NUM_BEAMS", "4"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("M2_NO_REPEAT_NGRAM_SIZE", "2"))

_model = None
_processor = None
_tokenizer = None

ENVIRONMENT_TERMS = {
    "forest", "grass", "grassland", "field", "road", "roadside", "snow",
    "water", "river", "lake", "wetland", "tree", "branch", "vegetation",
    "ground", "urban", "rural", "mountain", "shore", "beach", "desert",
}

BEHAVIOR_TERMS = {
    "standing", "sitting", "walking", "running", "flying", "swimming",
    "feeding", "resting", "looking", "perched", "hunting", "foraging",
}

SCENE_TERMS = {
    "animal", "wildlife", "field observation", "habitat", "ecology",
    "environment", "natural context", "behavioral context",
}


def _load_captioner():
    global _model, _processor, _tokenizer

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"M2 model folder not found: {MODEL_PATH}")

        _model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _processor = ViTImageProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _model.to(DEVICE)
        _model.eval()

    return _model, _processor, _tokenizer


def _load_image(image_path):
    image = Image.open(image_path)
    return image.convert("RGB") if image.mode != "RGB" else image


def _normalize_caption(text):
    text = re.sub(r"\s+", " ", text.strip())
    return text + "." if text and not text.endswith(".") else text


def _unique_keep_order(values):
    out = []
    for value in values:
        value = value.strip()
        if value and value not in out:
            out.append(value)
    return out


def _extract_visual_context(caption):
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", caption.lower())

    environment = [token for token in tokens if token in ENVIRONMENT_TERMS]
    behavior = [token for token in tokens if token in BEHAVIOR_TERMS]

    scene = []
    for token in tokens:
        if token not in environment and token not in behavior:
            scene.append(token)

    environment = _unique_keep_order(environment)
    behavior = _unique_keep_order(behavior)
    scene = _unique_keep_order(scene[:8])

    terms = _unique_keep_order(environment + behavior + scene + list(SCENE_TERMS))[:18]

    visual_prompt_context = (
        "Use the visual caption only as environmental, behavioral, and scene context. "
        "Do not treat it as taxonomic evidence. "
        f"Caption: {caption}"
    )

    return {
        "environment_terms": environment,
        "behavior_terms": behavior,
        "scene_terms": scene,
        "visual_context_terms": terms,
        "visual_prompt_context": visual_prompt_context,
    }


def process(context):
    ensure_context(context)

    image_path = Path(context["request"]["image_path"])
    image_name = context["request"]["image_name"]

    if not image_path.exists():
        raise FileNotFoundError(f"M2 image not found: {image_path}")

    model, processor, tokenizer = _load_captioner()
    image = _load_image(image_path)

    pixel_values = processor(images=[image], return_tensors="pt").pixel_values.to(DEVICE)

    generation_kwargs = {
        "max_length": MAX_LENGTH,
        "num_beams": NUM_BEAMS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    }

    with torch.no_grad():
        output_ids = model.generate(pixel_values, **generation_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = _normalize_caption(caption)
    visual_context = _extract_visual_context(caption)

    return {
        "image_name": image_name,
        "model_path": str(MODEL_PATH),
        "device": DEVICE,
        "caption": caption,
        **visual_context,
        "query_expansion_hint": " ".join(visual_context["visual_context_terms"]),
        "generation_config": generation_kwargs,
        "notes": [
            "M2 does not consume M1 outputs.",
            "M2 is only auxiliary context for environment, behavior, and scene.",
            "M2 must not be used as species evidence.",
        ],
    }
