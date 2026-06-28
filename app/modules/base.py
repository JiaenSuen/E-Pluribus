REQUIRED_CONTEXT_KEYS = ("request", "outputs")


def ensure_context(context):
    missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"Missing context keys: {', '.join(missing)}")
