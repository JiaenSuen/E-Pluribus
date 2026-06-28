from ..base import ensure_context

MODULE_ID = "m4_retrieval"
MODULE_NAME = "M4 Hybrid Literature Retrieval"
MODULE_ORDER = 40
MODULE_SOURCE = "default"
MODULE_DESCRIPTION = "Placeholder BM25 plus embedding retrieval candidate generator."


def _compact_query(query, context_terms, species_candidates):
    species = " ".join(item["scientific_name"] for item in species_candidates[:2])
    terms = " ".join(context_terms[:4])
    return f"{query} {species} {terms}".strip()


def process(context):
    ensure_context(context)
    query = context["request"]["query"]
    m2 = context["outputs"].get("m2_context", {}).get("data", {})
    m3 = context["outputs"].get("m3_classifier", {}).get("data", {})
    context_terms = m2.get("visual_context_terms", [])
    species_candidates = m3.get("species_candidates", [])
    normalized_query = _compact_query(query, context_terms, species_candidates)

    documents = [
        {
            "id": "DOC-001",
            "title": "Habitat-linked bird occurrence patterns in fragmented vegetation",
            "abstract": "A placeholder candidate about bird habitat, field observations, and local vegetation structure.",
            "keywords": ["habitat", "vegetation", "bird ecology"],
            "source": "Local demo corpus",
            "year": 2024,
            "doi": "10.0000/demo.001",
            "retrieval_score": 0.82,
        },
        {
            "id": "DOC-002",
            "title": "Context-aware retrieval for ecological observation workflows",
            "abstract": "A placeholder candidate about combining visual context, query intent, and evidence selection.",
            "keywords": ["retrieval", "reranking", "ecological research"],
            "source": "Local demo corpus",
            "year": 2025,
            "doi": "10.0000/demo.002",
            "retrieval_score": 0.77,
        },
        {
            "id": "DOC-003",
            "title": "Species refinement under incremental taxonomic updates",
            "abstract": "A placeholder candidate about class incremental learning for fine-grained species recognition.",
            "keywords": ["CIL", "species recognition", "taxonomy"],
            "source": "Local demo corpus",
            "year": 2023,
            "doi": "10.0000/demo.003",
            "retrieval_score": 0.71,
        },
        {
            "id": "DOC-004",
            "title": "Field note integration for biodiversity knowledge systems",
            "abstract": "A placeholder candidate about local research memory, field notes, PDFs, and observation records.",
            "keywords": ["field notes", "local memory", "biodiversity"],
            "source": "Local demo corpus",
            "year": 2022,
            "doi": "10.0000/demo.004",
            "retrieval_score": 0.66,
        },
    ]

    return {
        "normalizer": "EcoText Normalizer placeholder",
        "retriever": "BM25 + sentence embedding placeholder",
        "normalized_query": normalized_query,
        "candidate_documents": documents,
    }
