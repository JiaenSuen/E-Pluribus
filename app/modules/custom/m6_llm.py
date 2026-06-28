import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests

from app.modules.base import ensure_context

MODULE_ID = "m6_llm"
MODULE_NAME = "M6 Evidence-based Research Synthesis"
MODULE_ORDER = 60
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "Answer original query using M2 context, M3 species, ADW supplement, and M5 literature evidence. No M1 input."

BASE_DIR = Path(__file__).resolve().parents[3]

OLLAMA_MODEL = os.getenv("M6_OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("M6_OLLAMA_BASE_URL", "")
OUTPUT_LANGUAGE = os.getenv("M6_OUTPUT_LANGUAGE", "Traditional Chinese")
MAX_ABSTRACT_CHARS = int(os.getenv("M6_MAX_ABSTRACT_CHARS", "1200"))
MAX_DOCUMENTS = int(os.getenv("M6_MAX_DOCUMENTS", "5"))
SAVE_LOG = os.getenv("M6_SAVE_LOG", "1") == "1"

USE_ADW = os.getenv("M6_USE_ADW", "1") == "1"
ADW_TIMEOUT = float(os.getenv("M6_ADW_TIMEOUT", "8"))
MAX_ADW_CHARS = int(os.getenv("M6_MAX_ADW_CHARS", "2200"))

_llm = None


def _load_llm():
    global _llm

    if _llm is None:
        from langchain_ollama import OllamaLLM

        kwargs = {"model": OLLAMA_MODEL}
        if OLLAMA_BASE_URL:
            kwargs["base_url"] = OLLAMA_BASE_URL

        _llm = OllamaLLM(**kwargs)

    return _llm


def _truncate(text, max_chars):
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.I)
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _get_primary_species(context):
    m3_data = context["outputs"].get("m3_classifier", {}).get("data", {})
    primary = m3_data.get("primary_species")
    if primary and primary.get("scientific_name"):
        return {
            "scientific_name": primary.get("scientific_name"),
            "common_name": primary.get("common_name"),
            "confidence": primary.get("confidence"),
        }

    candidates = m3_data.get("species_candidates", [])
    if not candidates:
        return None

    top = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    return {
        "scientific_name": top.get("scientific_name"),
        "common_name": top.get("common_name"),
        "confidence": top.get("confidence"),
    }


def _collect_visual_prompt_context(context):
    m2_data = context["outputs"].get("m2_context", {}).get("data", {})

    return {
        "caption": m2_data.get("caption", ""),
        "environment_terms": m2_data.get("environment_terms", [])[:12],
        "behavior_terms": m2_data.get("behavior_terms", [])[:12],
        "scene_terms": m2_data.get("scene_terms", [])[:10],
        "visual_prompt_context": m2_data.get("visual_prompt_context", ""),
        "usage_rule": "Use only as environment, behavior, and scene context. Do not use as taxonomic proof.",
    }


def _collect_top_documents(context):
    m5_data = context["outputs"].get("m5_reranker", {}).get("data", {})
    documents = m5_data.get("top_documents", [])

    if not documents:
        m4_data = context["outputs"].get("m4_retrieval", {}).get("data", {})
        documents = m4_data.get("candidate_documents", [])[:MAX_DOCUMENTS]

    compact_docs = []
    for index, doc in enumerate(documents[:MAX_DOCUMENTS], start=1):
        compact_docs.append(
            {
                "rank": index,
                "title": doc.get("title", ""),
                "abstract": _truncate(doc.get("abstract", ""), MAX_ABSTRACT_CHARS),
                "year": doc.get("year", ""),
                "source": doc.get("source", ""),
                "doi": doc.get("doi", ""),
                "url": doc.get("url", ""),
                "contextual_relevance": doc.get("contextual_relevance"),
                "retrieval_score": doc.get("retrieval_score"),
                "matched_species_terms": doc.get("matched_species_terms", []),
                "matched_keywords": doc.get("matched_keywords", []),
            }
        )

    return compact_docs


def _is_identity_query(query):
    q = query.lower().strip()
    patterns = [
        "what is it", "what animal", "what species", "identify", "id this",
        "這是什麼", "這是甚麼", "是什麼", "是甚麼", "這是誰", "物種是",
    ]
    return any(pattern in q for pattern in patterns)


def _format_documents(documents):
    if not documents:
        return "No selected literature documents were available."

    blocks = []
    for doc in documents:
        blocks.append(
            f"""
Document {doc['rank']}
Title: {doc['title']}
Year: {doc['year']}
Source: {doc['source']}
DOI: {doc['doi']}
Contextual relevance: {doc['contextual_relevance']}
Matched species terms: {', '.join(doc.get('matched_species_terms', []))}
Matched ecology keywords: {', '.join(doc.get('matched_keywords', []))}
Abstract:
{doc['abstract']}
""".strip()
        )

    return "\n\n".join(blocks)


def _adw_url_for_species(scientific_name):
    slug = re.sub(r"\s+", "_", scientific_name.strip())
    return f"https://animaldiversity.org/accounts/{slug}/"


def _extract_adw_section(raw_html, heading):
    pattern = rf"<h[23][^>]*>\s*{re.escape(heading)}\s*</h[23]>([\s\S]*?)(?=<h[23][^>]*>|</article>|</main>)"
    match = re.search(pattern, raw_html, flags=re.I)
    if not match:
        return ""
    return _truncate(_clean_text(match.group(1)), 700)


def _fetch_adw_species_profile(scientific_name):
    if not USE_ADW or not scientific_name:
        return None

    url = _adw_url_for_species(scientific_name)
    headers = {"User-Agent": "EcoSystemResearchPrototype/0.1 (+local research assistant)"}

    try:
        response = requests.get(url, headers=headers, timeout=ADW_TIMEOUT)
        if response.status_code != 200:
            return {"source": "Animal Diversity Web", "url": url, "available": False}
        raw = response.text
    except Exception as exc:
        return {"source": "Animal Diversity Web", "url": url, "available": False, "error": str(exc)}

    sections = {}
    for heading in [
        "Geographic Range", "Habitat", "Physical Description", "Development",
        "Reproduction", "Lifespan/Longevity", "Behavior", "Communication and Perception",
        "Food Habits", "Predation", "Ecosystem Roles", "Conservation Status",
    ]:
        value = _extract_adw_section(raw, heading)
        if value:
            sections[heading] = value

    if not sections:
        body = _truncate(_clean_text(raw), MAX_ADW_CHARS)
    else:
        body = _truncate("\n".join(f"{k}: {v}" for k, v in sections.items()), MAX_ADW_CHARS)

    return {
        "source": "Animal Diversity Web",
        "url": url,
        "available": bool(body),
        "scientific_name": scientific_name,
        "sections": sections,
        "summary_text": body,
        "use_rule": "Use as natural-history background for the M3 species; still prioritize selected literature for research claims.",
    }


def _build_prompt(original_query, primary_species, visual_context, adw_profile, documents):
    docs_text = _format_documents(documents)
    identity_query = _is_identity_query(original_query)

    if identity_query:
        task_style = (
            "The user is asking a simple identification question. "
            "Give a clear and useful introduction to the likely species. "
            "Use literature only if it adds meaningful biological or ecological context. "
            "Do not force a paper-by-paper review."
        )
    else:
        task_style = (
            "The user is asking an ecological or scientific question. "
            "Answer the original question directly. "
            "Use the selected documents as the main evidence when they are relevant. "
            "Also judge whether the documents fit the user's scientific context."
        )

    prompt = f"""
You are an ecology and natural-history research assistant for researchers.

Respond in {OUTPUT_LANGUAGE}.

Your task:
Answer the user's original question directly and naturally.
Do not answer any expanded query.
Do not reveal search strings, retrieval prompts, model stages, scores, JSON, or implementation details.

Original user question:
{original_query}

User intent:
{task_style}

Likely species from image-based classification:
Use this as the main taxonomic hypothesis.
If confidence is available, express uncertainty appropriately.
{json.dumps(primary_species, ensure_ascii=False, indent=2)}

Scene and behavior context:
Use this only to describe possible environment, posture, activity, habitat, or visual setting.
Do not use it as taxonomic proof.
{json.dumps(visual_context, ensure_ascii=False, indent=2)}

Natural-history background:
Use this only if available and relevant.
It may support a general species introduction, but it should not replace selected literature when the user asks a research question.
{json.dumps(adw_profile, ensure_ascii=False, indent=2)}

Selected literature and document evidence:
Use these documents as the main research evidence when they are relevant to the original question.
If the documents are weak, generic, or mismatched, say so briefly and do not overuse them.
{docs_text}

Answering rules:
- Answer the original user question first.
- For "what is it" or simple identification questions, start with the likely species and provide a concise species introduction.
- For research questions, synthesize the selected literature and explain how well it fits the user's scientific context.
- Do not call the animal a dog unless the species hypothesis or evidence supports that.
- Do not invent authors, paper details, DOI, locations, methods, findings, or certainty.
- Do not discuss bounding boxes, object detection, crop generation, model pipeline, retrieval ranking, or internal modules.
- Do not output query strings or score formulas.
- Use scene context only for environment or behavior.
- Use the species hypothesis as tentative unless confidence is very high.
- If evidence is insufficient, say what is known, what is uncertain, and what additional evidence would be needed.

Output style:
Write only the final answer for the researcher.
Use clean paragraphs.
Avoid empty headings.
Avoid pipeline jargon.
Do not output JSON.
"""
    return prompt.strip()


def _fallback_answer(primary_species, visual_context, adw_profile, documents, error_message):
    name = "未知物種"
    confidence = None
    if primary_species:
        name = primary_species.get("scientific_name") or name
        confidence = primary_species.get("confidence")

    parts = [f"目前系統最可能辨識為 **{name}**。"]
    if confidence is not None:
        parts.append(f"M3 分類信心約為 {confidence}。")

    if adw_profile and adw_profile.get("summary_text"):
        parts.append("\n可用的自然史補充資料摘要：")
        parts.append(_truncate(adw_profile["summary_text"], 900))

    if visual_context.get("caption"):
        parts.append("\n影像情境可作為輔助參考：" + visual_context.get("caption", ""))

    if documents:
        titles = [doc.get("title", "") for doc in documents[:MAX_DOCUMENTS] if doc.get("title")]
        if titles:
            parts.append("\n目前可參考的文獻標題：")
            parts.extend([f"- {title}" for title in titles])

    parts.append(f"\n本地 Llama 呼叫失敗：{error_message}")
    return "\n".join(parts)


def _save_log(payload):
    if not SAVE_LOG:
        return None

    log_dir = BASE_DIR / "logs" / "m6"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return str(path)


def process(context):
    ensure_context(context)

    original_query = context["request"].get("query", "").strip()
    primary_species = _get_primary_species(context)
    visual_context = _collect_visual_prompt_context(context)
    documents = _collect_top_documents(context)

    scientific_name = primary_species.get("scientific_name") if primary_species else ""
    adw_profile = _fetch_adw_species_profile(scientific_name)

    prompt = _build_prompt(
        original_query=original_query,
        primary_species=primary_species,
        visual_context=visual_context,
        adw_profile=adw_profile,
        documents=documents,
    )

    error_message = None
    try:
        llm = _load_llm()
        answer = llm.invoke(prompt)
    except Exception as exc:
        error_message = str(exc)
        answer = _fallback_answer(
            primary_species=primary_species,
            visual_context=visual_context,
            adw_profile=adw_profile,
            documents=documents,
            error_message=error_message,
        )

    evidence_package = {
        "primary_species": primary_species,
        "visual_prompt_context": visual_context,
        "animal_diversity_web": adw_profile,
        "top_documents": documents,
        "llm_model": OLLAMA_MODEL,
        "llm_error": error_message,
        "excluded_context": ["M1 detections", "M1 bounding boxes", "M1 detector labels"],
    }

    log_path = _save_log(
        {
            "prompt": prompt,
            "answer": answer,
            "evidence_package": evidence_package,
        }
    )

    return {
        "model": OLLAMA_MODEL,
        "answer": answer,
        "evidence_package": evidence_package,
        "recommended_next_steps": [],
        "uncertainty": [],
        "log_path": log_path,
    }
