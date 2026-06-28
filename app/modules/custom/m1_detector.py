import os
from pathlib import Path

from ultralytics import YOLO

from app.modules.base import ensure_context

MODULE_ID = "m1_detector"
MODULE_NAME = "M1 Ultralytics Visual Detection"
MODULE_ORDER = 10
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "Ultralytics YOLO-based genus/coarse ecological object detector."

BASE_DIR = Path(__file__).resolve().parents[3]

WEIGHTS_PATH = Path(
    os.getenv("M1_WEIGHTS", BASE_DIR / "model_weights" / "m1_yolo" / "best.pt")
)

CONF_THRESHOLD = float(os.getenv("M1_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("M1_IOU", "0.70"))
IMAGE_SIZE = int(os.getenv("M1_IMGSZ", "640"))
DEVICE = os.getenv("M1_DEVICE", "")  # "", "cpu", "0", "cuda:0"
MAX_DETECTIONS = int(os.getenv("M1_MAX_DETECTIONS", "50"))

_model = None


def _load_model():
    global _model

    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"M1 weights not found: {WEIGHTS_PATH}")
        _model = YOLO(str(WEIGHTS_PATH))

    return _model


def _to_float_list(values, digits=6):
    return [round(float(v), digits) for v in values]


def process(context):
    ensure_context(context)

    image_path = context["request"]["image_path"]
    image_name = context["request"]["image_name"]

    model = _load_model()

    predict_kwargs = {
        "source": image_path,
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "imgsz": IMAGE_SIZE,
        "max_det": MAX_DETECTIONS,
        "verbose": False,
    }

    if DEVICE:
        predict_kwargs["device"] = DEVICE

    results = model.predict(**predict_kwargs)

    if not results:
        return {
            "image_name": image_name,
            "weights": str(WEIGHTS_PATH),
            "detections": [],
            "feature_summary": {
                "regions_found": 0,
                "dominant_context": "no detection",
            },
        }

    result = results[0]
    names = result.names or getattr(model, "names", {})

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.cpu()

        xyxy_abs = boxes.xyxy.tolist()
        xywh_norm = boxes.xywhn.tolist()
        xyxy_norm = boxes.xyxyn.tolist()
        confs = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()

        for index, class_id in enumerate(class_ids):
            class_id = int(class_id)
            label = names.get(class_id, str(class_id))

            detections.append(
                {
                    "label": label,
                    "genus": label,
                    "class_id": class_id,
                    "confidence": round(float(confs[index]), 6),
                    "bbox_xyxy_abs": _to_float_list(xyxy_abs[index], digits=2),
                    "bbox_xyxy_norm": _to_float_list(xyxy_norm[index]),
                    "bbox_xywh_norm": _to_float_list(xywh_norm[index]),
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "image_name": image_name,
        "weights": str(WEIGHTS_PATH),
        "framework": "ultralytics",
        "task": "object_detection",
        "image_shape": {
            "height": int(result.orig_shape[0]),
            "width": int(result.orig_shape[1]),
        },
        "thresholds": {
            "confidence": CONF_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "max_detections": MAX_DETECTIONS,
        },
        "detections": detections,
        "feature_summary": {
            "regions_found": len(detections),
            "top_genus": detections[0]["genus"] if detections else None,
            "top_confidence": detections[0]["confidence"] if detections else None,
        },
    }