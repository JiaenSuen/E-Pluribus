import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.modules.base import ensure_context

MODULE_ID = "m3_classifier"
MODULE_NAME = "M3 EConvNeXt Crop Species Classifier"
MODULE_ORDER = 30
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "M1-crop species refinement. Downstream query uses only the highest-confidence species."

BASE_DIR = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
M3_DIR = MODULE_DIR / "M3"

if str(M3_DIR) not in sys.path:
    sys.path.insert(0, str(M3_DIR))

from app.modules.custom.M3.EConvNeXt import (  # noqa: E402
    build_EConvNeXt_Mini,
    build_EConvNeXt_Tiny,
    build_EConvNeXt_Small,
)


def _resolve_path(env_name, default_path):
    value = os.getenv(env_name)
    path = Path(value) if value else Path(default_path)
    return path if path.is_absolute() else BASE_DIR / path


MODEL_VARIANT = os.getenv("M3_MODEL_VARIANT", "mini").lower()
PTH_PATH = _resolve_path(
    "M3_PTH_PATH",
    BASE_DIR / "model_weights" / "m3_econvnext" / "FoxSpecies_econvnext_mini_best.pth",
)
CLASS_NAMES_PATH = _resolve_path("M3_CLASS_NAMES_PATH", M3_DIR / "classes.txt")
DEVICE = os.getenv("M3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = int(os.getenv("M3_IMG_SIZE", "224"))
TOP_K = int(os.getenv("M3_TOP_K", "5"))
MAX_CROPS = int(os.getenv("M3_MAX_CROPS", "10"))
CROP_PADDING_RATIO = float(os.getenv("M3_CROP_PADDING_RATIO", "0.05"))
ROUTING_MODE = os.getenv("M3_ROUTING_MODE", "single")
FALLBACK_FULL_IMAGE = os.getenv("M3_FALLBACK_FULL_IMAGE", "0") == "1"

_model_cache = {}

CLASSIFIER_REGISTRY = {
    "default": {
        "genus": None,
        "model_variant": MODEL_VARIANT,
        "pth_path": PTH_PATH,
        "class_names_path": CLASS_NAMES_PATH,
    },
    # Future genus-specific classifiers can be added here.
}


def load_class_names(class_file_path):
    if not class_file_path.exists():
        raise FileNotFoundError(f"M3 class file not found: {class_file_path}")

    with open(class_file_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    if not class_names:
        raise ValueError(f"M3 class file is empty: {class_file_path}")

    return class_names


def build_model(model_variant, num_classes):
    model_variant = model_variant.lower()

    if model_variant == "mini":
        return build_EConvNeXt_Mini(num_classes=num_classes, img_channels=3)
    if model_variant == "tiny":
        return build_EConvNeXt_Tiny(num_classes=num_classes, img_channels=3)
    if model_variant == "small":
        return build_EConvNeXt_Small(num_classes=num_classes, img_channels=3)

    raise ValueError(f"Unsupported M3 model variant: {model_variant}. Available: mini, tiny, small.")


def load_checkpoint(model, pth_path, device):
    if not pth_path.exists():
        raise FileNotFoundError(f"M3 weight file not found: {pth_path}")

    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("Unsupported M3 checkpoint format.")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_classifier(classifier_id):
    if classifier_id in _model_cache:
        return _model_cache[classifier_id]

    config = CLASSIFIER_REGISTRY.get(classifier_id, CLASSIFIER_REGISTRY["default"])
    class_names = load_class_names(config["class_names_path"])
    model = build_model(config["model_variant"], len(class_names))
    model = load_checkpoint(model, config["pth_path"], DEVICE)
    model = model.to(DEVICE)
    model.eval()

    bundle = {
        "classifier_id": classifier_id if classifier_id in CLASSIFIER_REGISTRY else "default",
        "config": config,
        "model": model,
        "class_names": class_names,
        "transform": build_transform(),
    }

    _model_cache[bundle["classifier_id"]] = bundle
    return bundle


def _select_classifier_id(detection):
    if ROUTING_MODE != "by_genus":
        return "default"

    genus = detection.get("genus") or detection.get("label")
    return genus if genus in CLASSIFIER_REGISTRY else "default"


def _load_image(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"M3 image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def _bbox_to_xyxy_abs(detection, image_width, image_height):
    if detection.get("bbox_xyxy_abs"):
        x1, y1, x2, y2 = detection["bbox_xyxy_abs"]
        return float(x1), float(y1), float(x2), float(y2)

    if detection.get("bbox_xyxy_norm"):
        x1, y1, x2, y2 = detection["bbox_xyxy_norm"]
        return float(x1) * image_width, float(y1) * image_height, float(x2) * image_width, float(y2) * image_height

    if detection.get("bbox_xywh_norm"):
        x, y, w, h = detection["bbox_xywh_norm"]
        x = float(x) * image_width
        y = float(y) * image_height
        w = float(w) * image_width
        h = float(h) * image_height
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2

    return None


def _make_crop(image, detection):
    width, height = image.size
    bbox = _bbox_to_xyxy_abs(detection, width, height)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * CROP_PADDING_RATIO
    pad_y = box_height * CROP_PADDING_RATIO

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image.crop((x1, y1, x2, y2))
    crop_info = {
        "source": "m1_detection_crop",
        "bbox_xyxy_abs_used": [x1, y1, x2, y2],
        "m1_confidence": detection.get("confidence"),
    }
    return crop, crop_info


def _get_m1_detections(context):
    m1_data = context["outputs"].get("m1_detector", {}).get("data", {})
    detections = m1_data.get("detections", [])
    detections = sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return detections[:MAX_CROPS]


def _predict_crop(crop_image, bundle):
    image_tensor = bundle["transform"](crop_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = bundle["model"](image_tensor)
        probs = F.softmax(logits, dim=1)

    k = min(TOP_K, len(bundle["class_names"]))
    top_probs, top_indices = torch.topk(probs[0], k=k)

    candidates = []
    for rank, (prob, index) in enumerate(zip(top_probs, top_indices), start=1):
        class_id = int(index.item())
        confidence = float(prob.item())
        candidates.append(
            {
                "rank": rank,
                "class_id": class_id,
                "scientific_name": bundle["class_names"][class_id],
                "common_name": None,
                "confidence": round(confidence, 6),
            }
        )

    return candidates


def _select_primary_species(flattened_candidates):
    if not flattened_candidates:
        return None

    top = max(flattened_candidates, key=lambda item: float(item.get("confidence", 0.0)))
    return {
        "scientific_name": top.get("scientific_name"),
        "common_name": top.get("common_name"),
        "class_id": top.get("class_id"),
        "confidence": top.get("confidence"),
        "crop_index": top.get("crop_index"),
        "classifier_id": top.get("classifier_id"),
    }


def process(context):
    ensure_context(context)

    image_path = Path(context["request"]["image_path"])
    image_name = context["request"]["image_name"]
    image = _load_image(image_path)
    detections = _get_m1_detections(context)

    crop_predictions = []

    for crop_index, detection in enumerate(detections, start=1):
        crop_image, crop_info = _make_crop(image, detection)
        if crop_image is None:
            continue

        classifier_id = _select_classifier_id(detection)
        bundle = _load_classifier(classifier_id)
        candidates = _predict_crop(crop_image, bundle)

        crop_predictions.append(
            {
                "crop_index": crop_index,
                "input_mode": "m1_crop",
                "classifier_id": bundle["classifier_id"],
                "model_variant": bundle["config"]["model_variant"],
                "model_path": str(bundle["config"]["pth_path"]),
                "class_names_path": str(bundle["config"]["class_names_path"]),
                "crop_info": crop_info,
                "species_candidates": candidates,
                "top_prediction": candidates[0] if candidates else None,
            }
        )

    if not crop_predictions and FALLBACK_FULL_IMAGE:
        bundle = _load_classifier("default")
        candidates = _predict_crop(image, bundle)
        crop_predictions.append(
            {
                "crop_index": 1,
                "input_mode": "full_image_fallback",
                "classifier_id": "default",
                "model_variant": bundle["config"]["model_variant"],
                "model_path": str(bundle["config"]["pth_path"]),
                "class_names_path": str(bundle["config"]["class_names_path"]),
                "crop_info": None,
                "species_candidates": candidates,
                "top_prediction": candidates[0] if candidates else None,
            }
        )

    flattened_candidates = []
    for crop_result in crop_predictions:
        for candidate in crop_result["species_candidates"]:
            flattened_candidates.append(
                {
                    **candidate,
                    "crop_index": crop_result["crop_index"],
                    "classifier_id": crop_result["classifier_id"],
                    "input_mode": crop_result["input_mode"],
                }
            )

    flattened_candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    primary_species = _select_primary_species(flattened_candidates)

    query_species_terms = []
    if primary_species and primary_species.get("scientific_name"):
        query_species_terms.append(primary_species["scientific_name"])

    return {
        "image_name": image_name,
        "device": DEVICE,
        "routing_mode": ROUTING_MODE,
        "mvp_mode": ROUTING_MODE == "single",
        "future_multi_genus_ready": True,
        "num_m1_detections_received": len(detections),
        "num_crops_classified": len(crop_predictions),
        "active_classifiers": sorted({item["classifier_id"] for item in crop_predictions}),
        "primary_species": primary_species,
        "query_species_terms": query_species_terms,
        "species_candidates": flattened_candidates,
        "crop_predictions": crop_predictions,
        "cil_mode": "m1-crop species refinement",
        "notes": [
            "M1 is used only inside M3 for crop generation.",
            "Downstream query modules should use only primary_species / query_species_terms.",
            "Only the highest-confidence species is intended for M4 retrieval.",
        ],
        "metrics_stub": {
            "species_accuracy": None,
            "old_class_accuracy": None,
            "new_class_accuracy": None,
            "forgetting_measure": None,
        },
    }
