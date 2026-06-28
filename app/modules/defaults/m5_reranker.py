import re

from ..base import ensure_context

MODULE_ID = "m5_reranker"
MODULE_NAME = "M5 Context-aware EcoReranker"
MODULE_ORDER = 50
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder contextual reranker for research evidence selection."


def _tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower()))


def process(context):
    ensure_context(context)
    query = context["request"]["query"]
    m2 = context["outputs"].get("m2_context", {}).get("data", {})
    m4 = context["outputs"].get("m4_retrieval", {}).get("data", {})
    documents = m4.get("candidate_documents", [])

    context_text = " ".join([query, m2.get("caption", ""), " ".join(m2.get("visual_context_terms", []))])
    context_tokens = _tokens(context_text)
    ranked = []

    for doc in documents:
        doc_text = " ".join([doc.get("title", ""), doc.get("abstract", ""), " ".join(doc.get("keywords", []))])
        overlap = len(context_tokens & _tokens(doc_text))
        semantic_stub = min(0.95, doc.get("retrieval_score", 0.5) + overlap * 0.035)
        ranked.append(
            {
                **doc,
                "contextual_relevance": round(semantic_stub, 3),
                "rerank_reason": "Placeholder score using query, visual context, and keyword overlap.",
            }
        )

    ranked.sort(key=lambda item: item["contextual_relevance"], reverse=True)
    return {
        "reranker": "MiniLM cross-encoder placeholder",
        "score_range": "0-1",
        "top_documents": ranked[:3],
        "filtered_out": ranked[3:],
    }
