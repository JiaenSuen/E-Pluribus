from ..base import ensure_context

MODULE_ID = "m6_llm"
MODULE_NAME = "M6 Local LLM Synthesis"
MODULE_ORDER = 60
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder local LLM synthesis using online evidence and local research memory."


def process(context):
    ensure_context(context)
    query = context["request"]["query"]
    m1 = context["outputs"].get("m1_detector", {}).get("data", {})
    m2 = context["outputs"].get("m2_context", {}).get("data", {})
    m3 = context["outputs"].get("m3_classifier", {}).get("data", {})
    m5 = context["outputs"].get("m5_reranker", {}).get("data", {})

    detections = m1.get("detections", [])
    candidates = m3.get("species_candidates", [])
    top_docs = m5.get("top_documents", [])
    genus = detections[0].get("genus", "unknown") if detections else "unknown"
    species = candidates[0].get("scientific_name", "unknown") if candidates else "unknown"
    top_titles = [doc.get("title") for doc in top_docs]

    answer = (
        f"M1 detected a possible genus {genus} ecological goals, M2 creating image context ''{m2.get('caption', '')}'',"
        f"M3 provide a provisional species candidate : {species}。"
        "M4 establishes a candidate literature set based on query, image context, and species candidate; M5 reorders the literature based on contextual relevance."
        "M6 integrates top evidence with the local research memory interface into an evidence package that can be handed over to the local LLM."
        f"Regarding your query: {query}, we currently recommend checking whether the top-ranked documents are consistent with the species, habitat, behavior, or conservation intentions."
    )

    return {
        "model": "local Llama placeholder",
        "answer": answer,
        "evidence_package": {
            "visual_context": m2,
            "detections": detections,
            "species_candidates": candidates,
            "top_documents": top_docs,
            "local_memory_hits": [
                {
                    "id": "LM-001",
                    "source": "data/local_memory placeholder",
                    "summary": "No local index is connected yet. Add parser and embedding modules here.",
                }
            ],
            "top_document_titles": top_titles,
        },
        "recommended_next_steps": [
            "Replace M1-M3 placeholders with trained visual modules.",
            "Connect M4 to a real paper index or API-backed corpus.",
            "Train or plug in M5 as the main contextual relevance scorer.",
            "Connect M6 to a local LLM endpoint and Local Research Memory index.",
        ],
        "uncertainty": [
            "All model outputs are placeholders until custom modules are installed.",
            "Species-level classification should not be interpreted as scientific evidence in this prototype.",
        ],
    }
