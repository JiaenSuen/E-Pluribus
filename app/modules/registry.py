import importlib
import importlib.util
from pathlib import Path

DEFAULT_MODULES = [
    "app.modules.defaults.m1_detector",
    "app.modules.defaults.m2_context",
    "app.modules.defaults.m3_classifier",
    "app.modules.defaults.m4_retrieval",
    "app.modules.defaults.m5_reranker",
    "app.modules.defaults.m6_llm",
]

CUSTOM_DIR = Path(__file__).resolve().parent / "custom"


def _load_default_modules():
    return [importlib.import_module(path) for path in DEFAULT_MODULES]


def _load_custom_modules():
    modules = []
    for path in sorted(CUSTOM_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"custom_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "process"):
            modules.append(module)
    return modules


def load_pipeline_modules():
    by_id = {}
    for module in _load_default_modules() + _load_custom_modules():
        module_id = getattr(module, "MODULE_ID", module.__name__)
        by_id[module_id] = module
    return sorted(by_id.values(), key=lambda m: getattr(m, "MODULE_ORDER", 1000))


def get_module_manifest():
    manifest = []
    for module in load_pipeline_modules():
        manifest.append(
            {
                "id": getattr(module, "MODULE_ID", module.__name__),
                "name": getattr(module, "MODULE_NAME", module.__name__),
                "order": getattr(module, "MODULE_ORDER", 1000),
                "source": getattr(module, "MODULE_SOURCE", "custom"),
                "description": getattr(module, "MODULE_DESCRIPTION", ""),
            }
        )
    return manifest
