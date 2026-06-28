from ..base import ensure_context

MODULE_ID = "m1_detector"
MODULE_NAME = "M1 Visual Detection"
MODULE_ORDER = 10
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder genus-level detector with bounding boxes."


def process(context):
    ensure_context(context)
    image_name = context["request"]["image_name"]

    return {
        "image_name": image_name,
        "backbone": "E-ConvNeXt-Mini placeholder",
        "detector": "YOLOv10-compatible placeholder",
        "detections": [
            {
                "label": "ecological_object",
                "genus": "Vulpes",
                "confidence": 0.78,
                "bbox_xywh_norm": [0.21, 0.18, 0.48, 0.56],
            }
        ],
        "feature_summary": {
            "regions_found": 1,
            "dominant_context": "vegetated field observation",
        },
    }
