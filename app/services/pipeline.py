from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..modules.registry import load_pipeline_modules


def run_ecology_pipeline(image_path, image_url, query):
    request_id = uuid4().hex[:12]
    context = {
        "request": {
            "id": request_id,
            "query": query,
            "image_path": str(image_path),
            "image_url": image_url,
            "image_name": Path(image_path).name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "outputs": {},
    }

    trace = []
    for module in load_pipeline_modules():
        module_id = getattr(module, "MODULE_ID", module.__name__)
        module_name = getattr(module, "MODULE_NAME", module_id)
        started = datetime.now(timezone.utc)

        try:
            data = module.process(context)
            status = "ok"
            error = None
        except Exception as exc:  # Keep the prototype usable while modules evolve.
            data = {}
            status = "error"
            error = str(exc)

        finished = datetime.now(timezone.utc)
        elapsed_ms = round((finished - started).total_seconds() * 1000, 2)
        record = {
            "module_id": module_id,
            "module_name": module_name,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "data": data,
            "error": error,
        }
        context["outputs"][module_id] = record
        trace.append(record)

    llm_data = context["outputs"].get("m6_llm", {}).get("data", {})
    return {
        "request": context["request"],
        "final_answer": llm_data.get("answer", "No LLM output was produced."),
        "uncertainty": llm_data.get("uncertainty", []),
        "recommended_next_steps": llm_data.get("recommended_next_steps", []),
        "evidence_package": llm_data.get("evidence_package", {}),
        "trace": trace,
    }
