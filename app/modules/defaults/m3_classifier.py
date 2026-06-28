from ..base import ensure_context

MODULE_ID = "m3_classifier"
MODULE_NAME = "M3 Species Refinement and CIL"
MODULE_ORDER = 30
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder species-level classifier and CIL refinement stage."


def process(context):
    ensure_context(context)
    m1 = context["outputs"].get("m1_detector", {}).get("data", {})
    genus = "Turdus"
    if m1.get("detections"):
        genus = m1["detections"][0].get("genus", genus)

    return {
        "genus_route": genus,
        "cil_mode": "species-refinement placeholder",
        "species_candidates": [
            {
                "scientific_name": f"{genus} migratorius",
                "common_name": "American robin-like candidate",
                "confidence": 0.42,
            },
            {
                "scientific_name": f"{genus} merula",
                "common_name": "Blackbird-like candidate",
                "confidence": 0.31,
            },
        ],
        "metrics_stub": {
            "species_accuracy": None,
            "old_class_accuracy": None,
            "new_class_accuracy": None,
            "forgetting_measure": None,
        },
    }
