import math
import os
import re
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.modules.base import ensure_context

MODULE_ID = "m5_reranker"
MODULE_NAME = "M5 Species-focused CrossEncoder Reranker"
MODULE_ORDER = 50
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "Species-focused document reranker using a small CrossEncoder regression-style scorer. No FAISS and no original-query matching."

BASE_DIR = Path(__file__).resolve().parents[3]

CROSS_ENCODER_MODEL_PATH = Path(
    os.getenv(
        "M5_CROSS_ENCODER_MODEL_PATH",
        BASE_DIR / "model_weights" / "m5_reranker" / "ms-marco-MiniLM-L-6-v2",
    )
)

CROSS_ENCODER_HUB_MODEL = os.getenv(
    "M5_CROSS_ENCODER_HUB_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

DEVICE = os.getenv("M5_DEVICE", os.getenv("M4_DEVICE", "cpu"))
TOP_K = int(os.getenv("M5_TOP_K", "5"))
BATCH_SIZE = int(os.getenv("M5_BATCH_SIZE", "8"))
ALLOW_HUB_DOWNLOAD = os.getenv("M5_ALLOW_HUB_DOWNLOAD", "0") == "1"

CROSS_ENCODER_WEIGHT = float(os.getenv("M5_CROSS_ENCODER_WEIGHT", "0.75"))
SPECIES_TEXT_MATCH_WEIGHT = float(os.getenv("M5_SPECIES_TEXT_MATCH_WEIGHT", "0.20"))
M4_PRIOR_WEIGHT = float(os.getenv("M5_M4_PRIOR_WEIGHT", "0.05"))

MAX_DOC_CHARS = int(os.getenv("M5_MAX_DOC_CHARS", "2200"))
MIN_RELEVANCE = float(os.getenv("M5_MIN_RELEVANCE", "0.0"))

_model = None


ECOLOGY_HINTS = [
    "taxonomy", "distribution", "geographic range", "habitat",
    "behavior", "behaviour", "diet", "food habits", "reproduction",
    "breeding", "population", "ecology", "conservation", "natural history",
]


def _resolve_model_source():
    if CROSS_ENCODER_MODEL_PATH.exists():
        return str(CROSS_ENCODER_MODEL_PATH)

    if ALLOW_HUB_DOWNLOAD:
        return CROSS_ENCODER_HUB_MODEL

    raise FileNotFoundError(
        "M5 CrossEncoder model not found. "
        f"Expected local folder: {CROSS_ENCODER_MODEL_PATH}. "
        "Set M5_ALLOW_HUB_DOWNLOAD=1 to allow Hugging Face download, "
        "or download the model into model_weights/m5_reranker/."
    )


def _load_reranker():
    global _model

    if _model is None:
        _model = CrossEncoder(
            _resolve_model_source(),
            device=DEVICE,
        )

    return _model


def _clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value):
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _doc_text(doc):
    parts = [
        doc.get("title", ""),
        doc.get("abstract", ""),
        " ".join(doc.get("keywords", []) or []),
        str(doc.get("year", "")),
        doc.get("source", ""),
        doc.get("doi", ""),
    ]

    text = _clean_text(" ".join(parts))
    return text[:MAX_DOC_CHARS]


def _get_primary_species_payload(context):
    m3_data = context["outputs"].get("m3_classifier", {}).get("data", {})

    primary = m3_data.get("primary_species")
    if isinstance(primary, dict) and primary.get("scientific_name"):
        return {
            "scientific_name": _clean_text(primary.get("scientific_name", "")),
            "confidence": _safe_float(primary.get("confidence"), 0.0),
            "source": "m3_primary_species",
            "raw": primary,
        }

    top_prediction = m3_data.get("top_prediction")
    if isinstance(top_prediction, dict) and top_prediction.get("scientific_name"):
        return {
            "scientific_name": _clean_text(top_prediction.get("scientific_name", "")),
            "confidence": _safe_float(top_prediction.get("confidence"), 0.0),
            "source": "m3_top_prediction",
            "raw": top_prediction,
        }

    candidates = m3_data.get("species_candidates", []) or []
    candidates = [item for item in candidates if item.get("scientific_name")]

    if not candidates:
        return {
            "scientific_name": "",
            "confidence": 0.0,
            "source": "missing_m3_species",
            "raw": None,
        }

    best = max(candidates, key=lambda item: _safe_float(item.get("confidence"), 0.0))

    return {
        "scientific_name": _clean_text(best.get("scientific_name", "")),
        "confidence": _safe_float(best.get("confidence"), 0.0),
        "source": "m3_best_candidate",
        "raw": best,
    }


def _species_tokens(scientific_name):
    name = _clean_text(scientific_name)
    parts = [part for part in name.split() if part]

    genus = parts[0] if parts else ""
    species_epithet = parts[1] if len(parts) >= 2 else ""

    return {
        "scientific_name": name,
        "genus": genus,
        "species_epithet": species_epithet,
    }


def _build_species_focus_query(primary_species):
    species = primary_species.get("scientific_name", "")
    tokens = _species_tokens(species)
    genus = tokens["genus"]

    if species:
        focus = species
    elif genus:
        focus = genus
    else:
        focus = "unknown species"

    return _clean_text(
        " ".join(
            [
                focus,
                focus,
                genus,
                "taxonomy distribution habitat behavior diet ecology conservation natural history",
            ]
        )
    )


def _species_text_match(doc_text, tokens):
    text = doc_text.lower()
    scientific_name = tokens["scientific_name"].lower()
    genus = tokens["genus"].lower()
    species_epithet = tokens["species_epithet"].lower()

    score = 0.0
    matched = []

    if scientific_name and scientific_name in text:
        score += 1.0
        matched.append(tokens["scientific_name"])

    if genus and re.search(rf"\b{re.escape(genus)}\b", text):
        score += 0.35
        matched.append(tokens["genus"])

    if species_epithet and re.search(rf"\b{re.escape(species_epithet)}\b", text):
        score += 0.25
        matched.append(tokens["species_epithet"])

    hint_hits = []
    for hint in ECOLOGY_HINTS:
        if hint in text:
            hint_hits.append(hint)

    if hint_hits:
        score += min(0.20, 0.025 * len(hint_hits))

    return _clip01(score), sorted(set(matched)), hint_hits


def _m4_prior(doc):
    available = []

    for key in ["species_match_score", "retrieval_score", "bm25_score_norm", "semantic_score_norm"]:
        value = doc.get(key)
        if value is not None:
            available.append(_clip01(_safe_float(value, 0.0)))

    if not available:
        return 0.0

    return sum(available) / len(available)


def _predict_relevance(model, species_query, texts):
    pairs = [(species_query, text) for text in texts]
    raw_scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

    relevance = []
    for score in raw_scores:
        if isinstance(score, (list, tuple)):
            score = score[0]
        relevance.append(_sigmoid(float(score)))

    return relevance, raw_scores


def process(context):
    ensure_context(context)

    m4_data = context["outputs"].get("m4_retrieval", {}).get("data", {})
    documents = m4_data.get("candidate_documents", []) or []

    primary_species = _get_primary_species_payload(context)
    species_query = _build_species_focus_query(primary_species)
    species_token_pack = _species_tokens(primary_species.get("scientific_name", ""))

    if not documents:
        return {
            "reranker": "species-focused CrossEncoder reranker",
            "top_k": TOP_K,
            "primary_species": primary_species,
            "species_focus_query": species_query,
            "top_documents": [],
            "filtered_out": [],
            "notes": ["No M4 candidate documents were available."],
        }

    if not primary_species.get("scientific_name"):
        return {
            "reranker": "species-focused CrossEncoder reranker",
            "top_k": TOP_K,
            "primary_species": primary_species,
            "species_focus_query": species_query,
            "top_documents": [],
            "filtered_out": documents,
            "notes": [
                "M5 requires M3 species classification output.",
                "No primary species was available, so species-focused reranking was skipped.",
            ],
        }

    texts = [doc.get("text_for_rerank") or _doc_text(doc) for doc in documents]
    texts = [_clean_text(text)[:MAX_DOC_CHARS] for text in texts]

    model = _load_reranker()
    regression_scores, raw_scores = _predict_relevance(model, species_query, texts)

    rescored = []

    for index, doc in enumerate(documents):
        text = texts[index]
        species_match, matched_terms, ecology_hint_hits = _species_text_match(text, species_token_pack)
        prior = _m4_prior(doc)

        relevance = (
            CROSS_ENCODER_WEIGHT * float(regression_scores[index])
            + SPECIES_TEXT_MATCH_WEIGHT * float(species_match)
            + M4_PRIOR_WEIGHT * float(prior)
        )
        relevance = _clip01(relevance)

        if relevance < MIN_RELEVANCE:
            continue

        rescored.append(
            {
                "doc_index": index,
                "score": relevance,
                "cross_encoder_score_0_1": _clip01(regression_scores[index]),
                "cross_encoder_raw_score": float(raw_scores[index][0] if isinstance(raw_scores[index], (list, tuple)) else raw_scores[index]),
                "species_text_match_score": species_match,
                "m4_prior_score": prior,
                "matched_species_terms": matched_terms,
                "matched_ecology_context_terms": ecology_hint_hits,
            }
        )

    rescored.sort(key=lambda item: item["score"], reverse=True)

    k = min(TOP_K, len(rescored))
    selected = rescored[:k]
    selected_indices = {item["doc_index"] for item in selected}

    top_documents = []
    for rank, item in enumerate(selected, start=1):
        doc = documents[item["doc_index"]]
        top_documents.append(
            {
                **doc,
                "rank": rank,
                "m5_backend": "CrossEncoder regression-style species relevance scorer",
                "m5_model_path": str(CROSS_ENCODER_MODEL_PATH),
                "m5_primary_species": primary_species.get("scientific_name", ""),
                "m5_primary_species_confidence": primary_species.get("confidence", 0.0),
                "m5_species_relevance": round(float(item["score"]), 6),
                "m5_cross_encoder_score_0_1": round(float(item["cross_encoder_score_0_1"]), 6),
                "m5_cross_encoder_raw_score": round(float(item["cross_encoder_raw_score"]), 6),
                "m5_species_text_match_score": round(float(item["species_text_match_score"]), 6),
                "m5_m4_prior_score": round(float(item["m4_prior_score"]), 6),
                "matched_species_terms": item["matched_species_terms"],
                "matched_ecology_context_terms": item["matched_ecology_context_terms"],
                "contextual_relevance": round(float(item["score"]), 6),
                "rerank_reason": (
                    "M5 ranked this document by abstract/title relevance to the M3 primary species. "
                    "Original query, M1 outputs, M2 visual context, and FAISS are not used in this M5 version."
                ),
            }
        )

    filtered_out = [doc for index, doc in enumerate(documents) if index not in selected_indices]

    return {
        "reranker": "species-focused CrossEncoder reranker",
        "backend": "sentence-transformers CrossEncoder",
        "model_path": str(CROSS_ENCODER_MODEL_PATH),
        "hub_model": CROSS_ENCODER_HUB_MODEL if ALLOW_HUB_DOWNLOAD else None,
        "device": DEVICE,
        "top_k": TOP_K,
        "primary_species": primary_species,
        "species_focus_query": species_query,
        "score_weights": {
            "cross_encoder_regression": CROSS_ENCODER_WEIGHT,
            "species_text_match": SPECIES_TEXT_MATCH_WEIGHT,
            "m4_prior": M4_PRIOR_WEIGHT,
        },
        "top_documents": top_documents,
        "filtered_out": filtered_out,
        "notes": [
            "M5 intentionally ignores M1 outputs.",
            "M5 intentionally ignores the original user query.",
            "M5 intentionally ignores M2 visual context.",
            "M5 ranks documents by relevance to the M3 primary species only.",
            "Scores are normalized to 0-1 and can be replaced by a trained EcoReranker regression head later.",
        ],
    }
