import html
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from app.modules.base import ensure_context

MODULE_ID = "m4_retrieval"
MODULE_NAME = "M4 Species-centered Context Engineering Retrieval"
MODULE_ORDER = 40
MODULE_SOURCE = "custom"
MODULE_DESCRIPTION = "High-recall species-centered retrieval over literature APIs and biodiversity knowledge sources."

BASE_DIR = Path(__file__).resolve().parents[3]

EMBEDDING_MODEL_PATH = Path(
    os.getenv(
        "M4_EMBEDDING_MODEL_PATH",
        BASE_DIR / "model_weights" / "m4_embedding" / "all-MiniLM-L6-v2",
    )
)

DEVICE = os.getenv("M4_DEVICE", "cpu")
REQUEST_TIMEOUT = float(os.getenv("M4_REQUEST_TIMEOUT", "6"))
REQUEST_SLEEP = float(os.getenv("M4_REQUEST_SLEEP", "0.05"))
MAX_SECONDS = float(os.getenv("M4_MAX_SECONDS", "28"))
SOURCE_LIMIT = int(os.getenv("M4_SOURCE_LIMIT", "6"))
CANDIDATE_LIMIT = int(os.getenv("M4_CANDIDATE_LIMIT", "30"))
QUERY_VARIANTS_LIMIT = int(os.getenv("M4_QUERY_VARIANTS_LIMIT", "3"))

FALLBACK_TO_ORIGINAL_QUERY = os.getenv("M4_FALLBACK_TO_ORIGINAL_QUERY", "0") == "1"
STRICT_SPECIES_FILTER = os.getenv("M4_STRICT_SPECIES_FILTER", "1") == "1"

BM25_WEIGHT = float(os.getenv("M4_BM25_WEIGHT", "0.32"))
EMBEDDING_WEIGHT = float(os.getenv("M4_EMBEDDING_WEIGHT", "0.32"))
SPECIES_MATCH_WEIGHT = float(os.getenv("M4_SPECIES_MATCH_WEIGHT", "0.24"))
SOURCE_QUALITY_WEIGHT = float(os.getenv("M4_SOURCE_QUALITY_WEIGHT", "0.12"))

BM25_K1 = float(os.getenv("M4_BM25_K1", "1.5"))
BM25_B = float(os.getenv("M4_BM25_B", "0.75"))

NCBI_TOOL = os.getenv("M4_NCBI_TOOL", "EcoSystemM4")
NCBI_EMAIL = os.getenv("M4_NCBI_EMAIL", "")
BHL_API_KEY = os.getenv("M4_BHL_API_KEY", "")
USER_AGENT = os.getenv(
    "M4_USER_AGENT",
    "EcoSystemM4/0.1 (research prototype; contact=local)",
)

INCLUDE_SEMANTIC_SCHOLAR = os.getenv("M4_INCLUDE_SEMANTIC_SCHOLAR", "1") == "1"
INCLUDE_OPENALEX = os.getenv("M4_INCLUDE_OPENALEX", "1") == "1"
INCLUDE_CROSSREF = os.getenv("M4_INCLUDE_CROSSREF", "1") == "1"
INCLUDE_EUROPEPMC = os.getenv("M4_INCLUDE_EUROPEPMC", "1") == "1"
INCLUDE_PUBMED = os.getenv("M4_INCLUDE_PUBMED", "1") == "1"
INCLUDE_PMC = os.getenv("M4_INCLUDE_PMC", "1") == "1"
INCLUDE_DATACITE = os.getenv("M4_INCLUDE_DATACITE", "1") == "1"
INCLUDE_BHL = os.getenv("M4_INCLUDE_BHL", "1") == "1"
INCLUDE_GBIF = os.getenv("M4_INCLUDE_GBIF", "1") == "1"
INCLUDE_INATURALIST = os.getenv("M4_INCLUDE_INATURALIST", "1") == "1"
INCLUDE_ADW = os.getenv("M4_INCLUDE_ADW", "1") == "1"
INCLUDE_ARXIV = os.getenv("M4_INCLUDE_ARXIV", "0") == "1"
INCLUDE_SCIENCEDIRECT_TOPIC = os.getenv("M4_INCLUDE_SCIENCEDIRECT_TOPIC", "1") == "1"

ECOLOGY_TERMS = [
    "ecology", "natural history", "habitat", "distribution", "behavior",
    "behaviour", "diet", "feeding", "conservation", "biodiversity",
    "population", "reproduction", "breeding", "home range", "movement",
    "migration", "predation", "taxonomy", "occurrence", "wildlife",
]

STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "onto",
    "are", "was", "were", "has", "have", "had", "its", "their", "our",
    "use", "using", "based", "study", "studies", "paper", "research",
    "what", "which", "when", "where", "how", "why", "does", "did",
    "article", "journal", "species", "animal", "animals",
}

SOURCE_QUALITY = {
    "PMC": 0.96,
    "PubMed": 0.94,
    "EuropePMC": 0.92,
    "SemanticScholar": 0.90,
    "OpenAlex": 0.88,
    "CrossRef": 0.84,
    "DataCite": 0.80,
    "BHL": 0.78,
    "Nature/CrossRef": 0.88,
    "Wiley/CrossRef": 0.86,
    "ScienceDirect Topics": 0.58,
    "GBIF": 0.72,
    "iNaturalist": 0.62,
    "Animal Diversity Web": 0.70,
    "arXiv": 0.30,
}

DATABASE_SOURCES = {"GBIF", "iNaturalist", "Animal Diversity Web", "ScienceDirect Topics"}
LITERATURE_SOURCES = {
    "PMC", "PubMed", "EuropePMC", "SemanticScholar", "OpenAlex",
    "CrossRef", "Nature/CrossRef", "Wiley/CrossRef", "DataCite", "BHL", "arXiv",
}

_embed_model = None


def _load_embedder():
    global _embed_model
    if _embed_model is None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(f"M4 embedding model not found: {EMBEDDING_MODEL_PATH}")
        _embed_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device=DEVICE)
    return _embed_model


def _clean_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_text(v) for v in value if _clean_text(v)]
    if isinstance(value, tuple):
        return [_clean_text(v) for v in value if _clean_text(v)]
    text = _clean_text(value)
    return [text] if text else []


def _tokenize(text):
    text = _clean_text(text).lower()
    tokens = re.findall(r"[a-z][a-z\-]{2,}", text)
    return [token for token in tokens if token not in STOPWORDS]


def _requests_get(url, params=None, source_status=None, key="request"):
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html,*/*"},
        )
        response.raise_for_status()
        if REQUEST_SLEEP > 0:
            time.sleep(REQUEST_SLEEP)
        return response
    except Exception as exc:
        if source_status is not None:
            source_status[key] = {"status": "error", "error": str(exc)[:240]}
        return None


def _get_json(url, params=None, source_status=None, key="json"):
    response = _requests_get(url, params=params, source_status=source_status, key=key)
    if response is None:
        return {}
    try:
        return response.json()
    except Exception as exc:
        if source_status is not None:
            source_status[key] = {"status": "error", "error": f"Invalid JSON: {exc}"[:240]}
        return {}


def _get_text(url, params=None, source_status=None, key="text"):
    response = _requests_get(url, params=params, source_status=source_status, key=key)
    if response is None:
        return ""
    return response.text or ""


def _mark_status(source_status, key, count, note=""):
    source_status[key] = {"status": "ok", "count": int(count), "note": note}


def _time_left(start_time):
    return (time.time() - start_time) < MAX_SECONDS


def _paper_text(paper):
    return _clean_text(
        " ".join(
            [
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(_as_list(paper.get("keywords", []))),
                str(paper.get("year", "")),
                paper.get("doi", ""),
                paper.get("source", ""),
                paper.get("container_title", ""),
            ]
        )
    )


def _matched_ecology_terms(text):
    text_lower = text.lower()
    return [term for term in ECOLOGY_TERMS if term in text_lower]


def _split_species(scientific_name):
    parts = _clean_text(scientific_name).split()
    genus = parts[0] if parts else ""
    epithet = parts[1] if len(parts) > 1 else ""
    return genus, epithet


def _get_primary_species(context):
    m3_data = context.get("outputs", {}).get("m3_classifier", {}).get("data", {})

    primary = m3_data.get("primary_species") or m3_data.get("top_prediction")
    if primary and primary.get("scientific_name"):
        return {
            "scientific_name": _clean_text(primary.get("scientific_name")),
            "confidence": primary.get("confidence"),
            "source": "m3_primary_or_top_prediction",
        }

    candidates = m3_data.get("species_candidates", [])
    valid = [c for c in candidates if c.get("scientific_name")]
    if valid:
        top = max(valid, key=lambda item: float(item.get("confidence", 0.0)))
        return {
            "scientific_name": _clean_text(top.get("scientific_name")),
            "confidence": top.get("confidence"),
            "source": "m3_highest_confidence_candidate",
        }

    query = _clean_text(context.get("request", {}).get("query", ""))
    if FALLBACK_TO_ORIGINAL_QUERY and query:
        return {"scientific_name": query, "confidence": None, "source": "fallback_original_query"}

    return None


def _build_query_pack(context, taxon_context=None):
    original_query = _clean_text(context.get("request", {}).get("query", ""))
    primary_species = _get_primary_species(context)
    scientific_name = primary_species["scientific_name"] if primary_species else ""
    genus, epithet = _split_species(scientific_name)

    common_names = []
    if taxon_context:
        for key in ["common_names", "preferred_common_name"]:
            value = taxon_context.get(key)
            if isinstance(value, list):
                common_names.extend(value)
            elif value:
                common_names.append(value)
    common_names = list(dict.fromkeys([_clean_text(v) for v in common_names if _clean_text(v)]))[:5]

    base_terms = [scientific_name, genus]
    if common_names:
        base_terms.extend(common_names[:3])

    query_variants = []
    if scientific_name:
        query_variants.extend(
            [
                scientific_name,
                f"{scientific_name} ecology habitat distribution",
                f"{scientific_name} behavior diet conservation movement home range",
                f"{scientific_name} natural history reproduction population",
            ]
        )
    if common_names:
        query_variants.append(f"{scientific_name} {common_names[0]} ecology conservation")

    query_variants = [q for q in dict.fromkeys(_clean_text(q) for q in query_variants) if q]
    query_variants = query_variants[:QUERY_VARIANTS_LIMIT]

    scoring_query = _clean_text(
        " ".join(
            [scientific_name, scientific_name, genus, " ".join(common_names), " ".join(ECOLOGY_TERMS)]
        )
    )

    return {
        "original_query": original_query,
        "primary_species": primary_species,
        "scientific_name": scientific_name,
        "genus": genus,
        "epithet": epithet,
        "common_names": common_names,
        "query_variants": query_variants,
        "bm25_query": scoring_query,
        "embedding_query": scoring_query,
    }


def _make_doc(title, abstract, source, **kwargs):
    return {
        "title": _clean_text(title),
        "abstract": _clean_text(abstract),
        "keywords": _as_list(kwargs.get("keywords", [])),
        "year": kwargs.get("year", ""),
        "url": kwargs.get("url", ""),
        "doi": kwargs.get("doi", ""),
        "source": source,
        "container_title": _clean_text(kwargs.get("container_title", "")),
        "document_type": kwargs.get("document_type", "paper_metadata"),
        "source_quality": SOURCE_QUALITY.get(source, 0.5),
    }


def _search_semantic_scholar(query, source_status):
    key = f"SemanticScholar:{query}"
    data = _get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": SOURCE_LIMIT, "fields": "title,abstract,year,url,doi,fieldsOfStudy,venue"},
        source_status=source_status,
        key=key,
    ).get("data", [])
    docs = [
        _make_doc(
            p.get("title", ""),
            p.get("abstract", ""),
            "SemanticScholar",
            keywords=p.get("fieldsOfStudy", []) or [],
            year=p.get("year", ""),
            url=p.get("url", ""),
            doi=p.get("doi", ""),
            container_title=p.get("venue", ""),
        )
        for p in data
    ]
    _mark_status(source_status, key, len(docs))
    return docs


def _openalex_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    positions.sort(key=lambda item: item[0])
    return _clean_text(" ".join(word for _, word in positions))


def _search_openalex(query, source_status):
    key = f"OpenAlex:{query}"
    data = _get_json(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": SOURCE_LIMIT},
        source_status=source_status,
        key=key,
    )
    docs = []
    for p in data.get("results", []):
        concepts = p.get("concepts", []) or []
        keywords = [_clean_text(c.get("display_name", "")) for c in concepts if c.get("display_name")][:12]
        host = p.get("primary_location", {}) or {}
        source_info = host.get("source", {}) or {}
        docs.append(
            _make_doc(
                p.get("title", ""),
                _openalex_abstract(p.get("abstract_inverted_index")),
                "OpenAlex",
                keywords=keywords,
                year=p.get("publication_year", ""),
                url=p.get("id", ""),
                doi=(p.get("doi", "") or "").replace("https://doi.org/", ""),
                container_title=source_info.get("display_name", ""),
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs


def _search_crossref(query, source_status):
    key = f"CrossRef:{query}"
    data = _get_json(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": SOURCE_LIMIT, "select": "title,abstract,issued,URL,DOI,container-title,subject,publisher"},
        source_status=source_status,
        key=key,
    )
    docs = []
    for item in data.get("message", {}).get("items", []):
        year = ""
        date_parts = item.get("issued", {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
        container_title = (item.get("container-title") or [""])[0]
        publisher = _clean_text(item.get("publisher", ""))
        source = "CrossRef"
        joined = f"{container_title} {publisher}".lower()
        if "nature" in joined or "scientific reports" in joined:
            source = "Nature/CrossRef"
        elif "wiley" in joined or "journal of wildlife management" in joined:
            source = "Wiley/CrossRef"
        docs.append(
            _make_doc(
                (item.get("title") or [""])[0],
                item.get("abstract", ""),
                source,
                keywords=item.get("subject", []) or [],
                year=year,
                url=item.get("URL", ""),
                doi=item.get("DOI", ""),
                container_title=container_title,
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs


def _search_europepmc(query, source_status):
    key = f"EuropePMC:{query}"
    data = _get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f'"{query}"' if " " in query else query, "format": "json", "pageSize": SOURCE_LIMIT, "resultType": "core"},
        source_status=source_status,
        key=key,
    )
    docs = []
    for p in data.get("resultList", {}).get("result", []):
        url = ""
        if p.get("pmcid"):
            url = f"https://pmc.ncbi.nlm.nih.gov/articles/{p.get('pmcid')}/"
        elif p.get("doi"):
            url = f"https://doi.org/{p.get('doi')}"
        docs.append(
            _make_doc(
                p.get("title", ""),
                p.get("abstractText", ""),
                "EuropePMC",
                keywords=[],
                year=p.get("pubYear", ""),
                url=url,
                doi=p.get("doi", ""),
                container_title=p.get("journalTitle", ""),
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs


def _ncbi_params(extra=None):
    params = {"tool": NCBI_TOOL, "retmode": "json"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if extra:
        params.update(extra)
    return params


def _search_ncbi_ids(db, term, source_status):
    key = f"NCBI-{db}:esearch"
    data = _get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params=_ncbi_params({"db": db, "term": term, "retmax": SOURCE_LIMIT}),
        source_status=source_status,
        key=key,
    )
    return data.get("esearchresult", {}).get("idlist", [])


def _xml_text(node):
    if node is None:
        return ""
    return _clean_text(" ".join(node.itertext()))


def _search_pubmed(query_pack, source_status):
    scientific_name = query_pack["scientific_name"]
    if not scientific_name:
        return []
    term = f'"{scientific_name}"[Title/Abstract]'
    ids = _search_ncbi_ids("pubmed", term, source_status)
    if not ids:
        _mark_status(source_status, "PubMed", 0, "no ids")
        return []

    text = _get_text(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        source_status=source_status,
        key="PubMed:efetch",
    )
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        source_status["PubMed:efetch"] = {"status": "error", "error": "invalid xml"}
        return []

    docs = []
    for article in root.findall(".//PubmedArticle"):
        title = _xml_text(article.find(".//ArticleTitle"))
        abstract = _xml_text(article.find(".//Abstract"))
        journal = _xml_text(article.find(".//Journal/Title"))
        year = _xml_text(article.find(".//PubDate/Year"))
        doi = ""
        pmid = _xml_text(article.find(".//PMID"))
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = _xml_text(aid)
                break
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else "")
        docs.append(_make_doc(title, abstract, "PubMed", year=year, url=url, doi=doi, container_title=journal))
    _mark_status(source_status, "PubMed", len(docs))
    return docs


def _search_pmc(query_pack, source_status):
    scientific_name = query_pack["scientific_name"]
    if not scientific_name:
        return []
    term = f'"{scientific_name}"'
    ids = _search_ncbi_ids("pmc", term, source_status)
    if not ids:
        _mark_status(source_status, "PMC", 0, "no ids")
        return []

    text = _get_text(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pmc", "id": ",".join(ids), "retmode": "xml", "tool": NCBI_TOOL, "email": NCBI_EMAIL},
        source_status=source_status,
        key="PMC:efetch",
    )
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        source_status["PMC:efetch"] = {"status": "error", "error": "invalid xml"}
        return []

    docs = []
    for article in root.findall(".//article"):
        title = _xml_text(article.find(".//article-title"))
        abstract = _xml_text(article.find(".//abstract"))
        journal = _xml_text(article.find(".//journal-title"))
        year = _xml_text(article.find(".//pub-date/year"))
        doi = ""
        pmcid = ""
        for aid in article.findall(".//article-id"):
            if aid.attrib.get("pub-id-type") == "doi":
                doi = _xml_text(aid)
            if aid.attrib.get("pub-id-type") == "pmc":
                pmcid = _xml_text(aid)
        url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/" if pmcid and not pmcid.startswith("PMC") else f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else (f"https://doi.org/{doi}" if doi else "")
        docs.append(_make_doc(title, abstract, "PMC", year=year, url=url, doi=doi, container_title=journal))
    _mark_status(source_status, "PMC", len(docs))
    return docs


def _search_datacite(query, source_status):
    key = f"DataCite:{query}"
    data = _get_json(
        "https://api.datacite.org/dois",
        params={"query": query, "page[size]": SOURCE_LIMIT},
        source_status=source_status,
        key=key,
    )
    docs = []
    for item in data.get("data", []):
        attrs = item.get("attributes", {}) or {}
        titles = attrs.get("titles", []) or []
        descriptions = attrs.get("descriptions", []) or []
        title = titles[0].get("title", "") if titles else ""
        abstract = " ".join(d.get("description", "") for d in descriptions if d.get("description"))
        publisher = attrs.get("publisher", "")
        docs.append(
            _make_doc(
                title,
                abstract,
                "DataCite",
                keywords=attrs.get("subjects", []) or [],
                year=attrs.get("publicationYear", ""),
                url=attrs.get("url", ""),
                doi=attrs.get("doi", ""),
                container_title=publisher,
                document_type="repository_record",
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs


def _search_bhl(query_pack, source_status):
    scientific_name = query_pack["scientific_name"]
    if not scientific_name:
        return []
    params = {"op": "GetNameMetadata", "name": scientific_name, "format": "json"}
    if BHL_API_KEY:
        params["apikey"] = BHL_API_KEY
    key = "BHL:GetNameMetadata"
    data = _get_json("https://www.biodiversitylibrary.org/api3", params=params, source_status=source_status, key=key)
    result = data.get("Result", {}) or {}
    docs = []
    for title in result.get("Titles", []) or []:
        docs.append(
            _make_doc(
                title.get("FullTitle") or title.get("ShortTitle") or title.get("Title", ""),
                f"Biodiversity Heritage Library record associated with {scientific_name}.",
                "BHL",
                keywords=["biodiversity literature", "taxonomy", scientific_name],
                year=title.get("StartYear", ""),
                url=title.get("TitleUrl", ""),
                doi="",
                container_title="Biodiversity Heritage Library",
                document_type="biodiversity_literature_record",
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs[:SOURCE_LIMIT]


def _search_arxiv(query, source_status):
    key = f"arXiv:{query}"
    text = _get_text(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": SOURCE_LIMIT},
        source_status=source_status,
        key=key,
    )
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    docs = []
    for entry in root.findall("a:entry", ns):
        docs.append(
            _make_doc(
                _xml_text(entry.find("a:title", ns)),
                _xml_text(entry.find("a:summary", ns)),
                "arXiv",
                keywords=[],
                year="",
                url=_xml_text(entry.find("a:id", ns)),
                doi="arXiv",
            )
        )
    _mark_status(source_status, key, len(docs))
    return docs


def _get_gbif_context(scientific_name, source_status):
    if not scientific_name:
        return {}, []
    key = "GBIF:species_match"
    data = _get_json(
        "https://api.gbif.org/v1/species/match",
        params={"name": scientific_name},
        source_status=source_status,
        key=key,
    )
    if not data:
        return {}, []
    context = {
        "scientific_name": data.get("scientificName"),
        "canonical_name": data.get("canonicalName"),
        "rank": data.get("rank"),
        "status": data.get("status"),
        "confidence": data.get("confidence"),
        "usage_key": data.get("usageKey"),
        "kingdom": data.get("kingdom"),
        "phylum": data.get("phylum"),
        "class": data.get("class"),
        "order": data.get("order"),
        "family": data.get("family"),
        "genus": data.get("genus"),
    }
    abstract = " | ".join(f"{k}: {v}" for k, v in context.items() if v)
    doc = _make_doc(
        f"GBIF taxonomic backbone record for {scientific_name}",
        abstract,
        "GBIF",
        keywords=["taxonomy", "biodiversity", "species occurrence", scientific_name],
        year="",
        url=f"https://www.gbif.org/species/{data.get('usageKey')}" if data.get("usageKey") else "https://www.gbif.org/",
        document_type="species_database_record",
    )
    _mark_status(source_status, key, 1)
    return context, [doc]


def _get_inaturalist_context(scientific_name, source_status):
    if not scientific_name:
        return {}, []
    key = "iNaturalist:taxa"
    data = _get_json(
        "https://api.inaturalist.org/v1/taxa",
        params={"q": scientific_name, "rank": "species", "per_page": 3},
        source_status=source_status,
        key=key,
    )
    results = data.get("results", []) or []
    if not results:
        _mark_status(source_status, key, 0)
        return {}, []
    taxon = results[0]
    common = taxon.get("preferred_common_name", "")
    wikipedia_summary = taxon.get("wikipedia_summary", "") or ""
    context = {
        "preferred_common_name": common,
        "common_names": [common] if common else [],
        "taxon_id": taxon.get("id"),
        "name": taxon.get("name"),
        "rank": taxon.get("rank"),
    }
    doc = _make_doc(
        f"iNaturalist taxon profile for {taxon.get('name', scientific_name)}",
        wikipedia_summary or f"iNaturalist taxon record for {scientific_name}.",
        "iNaturalist",
        keywords=["citizen science", "observations", "taxonomy", scientific_name, common],
        year="",
        url=f"https://www.inaturalist.org/taxa/{taxon.get('id')}" if taxon.get("id") else "https://www.inaturalist.org/",
        document_type="species_database_record",
    )
    _mark_status(source_status, key, 1)
    return context, [doc]


def _get_adw_profile(scientific_name, source_status):
    genus, epithet = _split_species(scientific_name)
    if not genus or not epithet:
        return {}, []
    account = f"{genus}_{epithet}"
    url = f"https://animaldiversity.org/accounts/{account}/"
    key = "ADW:account"
    text = _get_text(url, source_status=source_status, key=key)
    if not text or "404" in text[:500].lower():
        _mark_status(source_status, key, 0)
        return {}, []

    plain = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    plain = re.sub(r"<style[\s\S]*?</style>", " ", plain, flags=re.I)
    headings = ["Geographic Range", "Habitat", "Physical Description", "Behavior", "Food Habits", "Conservation Status", "Reproduction"]
    sections = {}
    for heading in headings:
        pattern = rf"<h[23][^>]*>\s*{re.escape(heading)}\s*</h[23]>([\s\S]*?)(?=<h[23]|$)"
        match = re.search(pattern, plain, flags=re.I)
        if match:
            section_text = _clean_text(match.group(1))[:900]
            if section_text:
                sections[heading] = section_text
    abstract = " ".join(f"{k}: {v}" for k, v in sections.items())
    if not abstract:
        abstract = f"Animal Diversity Web species account for {scientific_name}."
    profile = {"url": url, "sections": sections}
    doc = _make_doc(
        f"Animal Diversity Web species account for {scientific_name}",
        abstract,
        "Animal Diversity Web",
        keywords=["natural history", "habitat", "behavior", "diet", "conservation", scientific_name],
        year="",
        url=url,
        document_type="species_account",
    )
    _mark_status(source_status, key, 1)
    return profile, [doc]


def _science_direct_topic_card(query_pack, source_status):
    names = query_pack.get("common_names", [])
    fallback = query_pack.get("genus", "")
    if not names and not fallback:
        return []
    topic = names[0] if names else fallback
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    if not slug:
        return []
    url = f"https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/{slug}"
    key = "ScienceDirectTopics:probe"
    response = _requests_get(url, source_status=source_status, key=key)
    if response is None or response.status_code >= 400:
        _mark_status(source_status, key, 0, "topic page unavailable")
        return []
    title = f"ScienceDirect Topics overview: {topic}"
    abstract = (
        f"ScienceDirect Topics page potentially relevant to {topic}. "
        "Use as broad background only; prefer peer-reviewed documents for research evidence."
    )
    _mark_status(source_status, key, 1)
    return [
        _make_doc(
            title,
            abstract,
            "ScienceDirect Topics",
            keywords=["topic overview", "biology", topic],
            url=url,
            document_type="encyclopedia_topic",
        )
    ]


def _deduplicate(docs):
    seen = set()
    unique = []
    for doc in docs:
        title_key = re.sub(r"\W+", "", doc.get("title", "").lower())[:140]
        doi_key = (doc.get("doi", "") or "").lower().replace("https://doi.org/", "")
        url_key = (doc.get("url", "") or "").lower().rstrip("/")
        key = doi_key or url_key or title_key
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _species_match_score(text, query_pack):
    text_lower = text.lower()
    scientific = query_pack.get("scientific_name", "").lower()
    genus = query_pack.get("genus", "").lower()
    epithet = query_pack.get("epithet", "").lower()
    common_names = [c.lower() for c in query_pack.get("common_names", [])]

    matched = []
    score = 0.0
    if scientific and scientific in text_lower:
        score += 1.0
        matched.append(query_pack["scientific_name"])
    elif genus and epithet and re.search(rf"\b{re.escape(genus)}\b", text_lower) and re.search(rf"\b{re.escape(epithet)}\b", text_lower):
        score += 0.85
        matched.append(query_pack["scientific_name"])
    elif genus and re.search(rf"\b{re.escape(genus)}\b", text_lower):
        score += 0.35
        matched.append(query_pack["genus"])

    for common in common_names:
        if common and common in text_lower:
            score += 0.45
            matched.append(common)
            break

    return min(score, 1.0), sorted(set(matched))


def _species_filter(doc, query_pack):
    if not STRICT_SPECIES_FILTER:
        return True
    if doc.get("source") in DATABASE_SOURCES:
        return True
    text = _paper_text(doc)
    score, _ = _species_match_score(text, query_pack)
    if score >= 0.35:
        return True
    ecology_terms = _matched_ecology_terms(text)
    return score > 0 and len(ecology_terms) >= 2


def _bm25_scores(query, documents):
    tokenized_docs = [_tokenize(doc) for doc in documents]
    query_tokens = _tokenize(query)
    if not tokenized_docs or not query_tokens:
        return np.zeros(len(documents), dtype=np.float32)

    doc_lengths = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
    avg_doc_length = float(np.mean(doc_lengths)) if len(doc_lengths) else 1.0
    doc_freq = Counter()
    term_freqs = []
    for doc_tokens in tokenized_docs:
        tf = Counter(doc_tokens)
        term_freqs.append(tf)
        for token in set(doc_tokens):
            doc_freq[token] += 1

    n_docs = len(tokenized_docs)
    scores = np.zeros(n_docs, dtype=np.float32)
    for token in query_tokens:
        if token not in doc_freq:
            continue
        df = doc_freq[token]
        idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
        for index, tf in enumerate(term_freqs):
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * (doc_lengths[index] / max(avg_doc_length, 1.0)))
            scores[index] += idf * ((freq * (BM25_K1 + 1)) / denominator)
    return scores


def _minmax(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _rank_candidates(query_pack, docs):
    if not docs:
        return []

    texts = [_paper_text(doc) for doc in docs]
    bm25_raw = _bm25_scores(query_pack["bm25_query"], texts)
    bm25_norm = _minmax(bm25_raw)

    model = _load_embedder()
    embeddings = model.encode(
        [query_pack["embedding_query"]] + texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    query_vec = embeddings[0]
    doc_vecs = embeddings[1:]
    semantic_raw = doc_vecs @ query_vec
    semantic_norm = (semantic_raw + 1.0) / 2.0

    ranked = []
    for index, doc in enumerate(docs):
        species_score, matched_species = _species_match_score(texts[index], query_pack)
        source_quality = float(doc.get("source_quality", SOURCE_QUALITY.get(doc.get("source", ""), 0.5)))
        final_score = (
            BM25_WEIGHT * float(bm25_norm[index])
            + EMBEDDING_WEIGHT * float(semantic_norm[index])
            + SPECIES_MATCH_WEIGHT * float(species_score)
            + SOURCE_QUALITY_WEIGHT * source_quality
        )
        ecology_terms = _matched_ecology_terms(texts[index])
        ranked.append(
            {
                **doc,
                "id": f"{doc.get('source', 'SRC')}-{index + 1:03d}",
                "retrieval_score": round(float(final_score), 6),
                "bm25_score_raw": round(float(bm25_raw[index]), 6),
                "bm25_score_norm": round(float(bm25_norm[index]), 6),
                "semantic_score_raw": round(float(semantic_raw[index]), 6),
                "semantic_score_norm": round(float(semantic_norm[index]), 6),
                "species_match_score": round(float(species_score), 6),
                "source_quality_score": round(source_quality, 6),
                "matched_species_terms": matched_species,
                "matched_keywords": ecology_terms,
                "ecology_keyword_score": len(ecology_terms),
                "text_for_rerank": texts[index][:4000],
                "retrieval_reason": (
                    "Species-centered context-engineering retrieval using exact species matching, "
                    "BM25, sentence embeddings, and source-quality priors."
                ),
            }
        )

    ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return ranked[:CANDIDATE_LIMIT]


def process(context):
    ensure_context(context)
    start_time = time.time()
    source_status = {}

    primary_species = _get_primary_species(context)
    if not primary_species:
        return {
            "retriever": "species-centered context-engineering retrieval",
            "candidate_documents": [],
            "source_status": {"M3": {"status": "error", "error": "No M3 primary species was available."}},
            "notes": ["M4 requires M3 species output. M1 and M2 are intentionally ignored."],
        }

    species_name = primary_species["scientific_name"]
    taxon_context = {}
    raw_docs = []

    if INCLUDE_GBIF and _time_left(start_time):
        gbif_context, docs = _get_gbif_context(species_name, source_status)
        taxon_context.update({k: v for k, v in gbif_context.items() if v})
        raw_docs.extend(docs)

    if INCLUDE_INATURALIST and _time_left(start_time):
        inat_context, docs = _get_inaturalist_context(species_name, source_status)
        for key, value in inat_context.items():
            if value and key not in taxon_context:
                taxon_context[key] = value
            elif key == "common_names" and value:
                taxon_context.setdefault("common_names", [])
                taxon_context["common_names"].extend(value)
        raw_docs.extend(docs)

    query_pack = _build_query_pack(context, taxon_context=taxon_context)

    if INCLUDE_ADW and _time_left(start_time):
        adw_profile, docs = _get_adw_profile(species_name, source_status)
        if adw_profile:
            taxon_context["adw_profile"] = adw_profile
        raw_docs.extend(docs)

    if INCLUDE_SCIENCEDIRECT_TOPIC and _time_left(start_time):
        raw_docs.extend(_science_direct_topic_card(query_pack, source_status))

    if INCLUDE_BHL and _time_left(start_time):
        raw_docs.extend(_search_bhl(query_pack, source_status))

    if INCLUDE_PUBMED and _time_left(start_time):
        raw_docs.extend(_search_pubmed(query_pack, source_status))

    if INCLUDE_PMC and _time_left(start_time):
        raw_docs.extend(_search_pmc(query_pack, source_status))

    source_searchers = []
    if INCLUDE_SEMANTIC_SCHOLAR:
        source_searchers.append(_search_semantic_scholar)
    if INCLUDE_OPENALEX:
        source_searchers.append(_search_openalex)
    if INCLUDE_CROSSREF:
        source_searchers.append(_search_crossref)
    if INCLUDE_EUROPEPMC:
        source_searchers.append(_search_europepmc)
    if INCLUDE_DATACITE:
        source_searchers.append(_search_datacite)
    if INCLUDE_ARXIV:
        source_searchers.append(_search_arxiv)

    for query in query_pack["query_variants"]:
        if not _time_left(start_time):
            break
        for searcher in source_searchers:
            if not _time_left(start_time):
                break
            try:
                raw_docs.extend(searcher(query, source_status))
            except Exception as exc:
                source_status[f"{searcher.__name__}:{query}"] = {"status": "error", "error": str(exc)[:240]}

    raw_docs = _deduplicate(raw_docs)
    species_filtered_docs = [doc for doc in raw_docs if _species_filter(doc, query_pack)]
    if not species_filtered_docs:
        species_filtered_docs = raw_docs

    candidates = _rank_candidates(query_pack, species_filtered_docs)

    return {
        "retriever": "species-centered context-engineering retrieval",
        "embedding_model_path": str(EMBEDDING_MODEL_PATH),
        "device": DEVICE,
        "original_query": query_pack["original_query"],
        "primary_species": primary_species,
        "scientific_name": query_pack["scientific_name"],
        "genus": query_pack["genus"],
        "common_names": query_pack["common_names"],
        "query_variants": query_pack["query_variants"],
        "bm25_query": query_pack["bm25_query"],
        "embedding_query": query_pack["embedding_query"],
        "taxon_context": taxon_context,
        "source_status": source_status,
        "source_limit": SOURCE_LIMIT,
        "candidate_limit": CANDIDATE_LIMIT,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "score_weights": {
            "bm25": BM25_WEIGHT,
            "embedding": EMBEDDING_WEIGHT,
            "species_match": SPECIES_MATCH_WEIGHT,
            "source_quality": SOURCE_QUALITY_WEIGHT,
        },
        "raw_document_count": len(raw_docs),
        "species_filtered_count": len(species_filtered_docs),
        "candidate_documents": candidates,
        "notes": [
            "M4 intentionally ignores M1 and M2 outputs.",
            "M4 uses only M3 primary species plus taxon-context enrichment.",
            "Database records are used as context cards; peer-reviewed literature should be prioritized by M5/M6 for research claims.",
        ],
    }
