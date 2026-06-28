from app.modules.base import ensure_context

MODULE_ID = "m2_context"
MODULE_NAME = "Custom Visual Context Module"
MODULE_ORDER = 20
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "Example override for the default M2 module."


def process(context):
    ensure_context(context)
    return {
        "caption": "Custom caption from your model.",
        "visual_context_terms": ["custom", "ecology", "observation"],
        "query_expansion_hint": "custom ecology observation",
    }
