from ..base import ensure_context

MODULE_ID = "m2_context"
MODULE_NAME = "M2 Visual Context Description"
MODULE_ORDER = 20
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder image-to-context caption module."


def process(context):
    ensure_context(context)
    m1 = context["outputs"].get("m1_detector", {}).get("data", {})
    genus = "unknown"
    if m1.get("detections"):
        genus = m1["detections"][0].get("genus", "unknown")

    return {
        "caption": "Field observation image with a likely biological subject in a vegetated habitat.",
        "visual_context_terms": ["field observation", "vegetation", "habitat", genus],
        "query_expansion_hint": f"{genus} habitat behavior conservation ecology",
    }
