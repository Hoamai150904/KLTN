import os
import json
import requests
import urllib3
from typing import Dict, Any, List, Tuple, Optional
from datetime import date, datetime

from dotenv import load_dotenv
from utils.es_http import request_kwargs
from sentence_transformers import SentenceTransformer
from utils.effective_filter import get_reference_date, build_effective_filter, _to_iso_date, DEFAULT_WEIGHTS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

ES_BASE_URL = (os.getenv("ES_BASE_URL") or os.getenv("ES_URL") or "").rstrip("/")
ES_INDEX = os.getenv("ES_INDEX")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
AUTH = (ES_USER, ES_PASSWORD) if (ES_USER and not ES_API_KEY) else None

VERIFY_TLS = os.getenv("ES_VERIFY_TLS", "false").lower() in ("1", "true", "yes", "y")
ES_VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "content_vector")

HF_MODEL = os.getenv("HF_MODEL", "BAAI/bge-m3")

MAX_TOKENS = int(os.getenv("EMB_MAX_TOKENS", "256"))

EMB_DIM = int(os.getenv("EMB_DIM", "1024"))

Q_PREFIX = os.getenv("Q_PREFIX", "Represent this sentence for searching relevant documents: ")
D_PREFIX = os.getenv("D_PREFIX", "Represent this document for retrieval: ")

device = "cuda" if os.getenv("FORCE_CPU", "0") not in ("1", "true", "yes") else "cpu"
print(f"HF_MODEL: {HF_MODEL}")

HF_TOKEN = os.getenv("HF_TOKEN", None)

if HF_TOKEN:
    model = SentenceTransformer(HF_MODEL, token=HF_TOKEN)
else:
    model = SentenceTransformer(HF_MODEL)

model.max_seq_length = MAX_TOKENS
print("Loaded SentenceTransformer. max_seq_length =", model.max_seq_length)


def embed_text(text: str, is_query: bool) -> Optional[List[float]]:
    text = (text or "").strip()
    if not text:
        return None

    if is_query:
        text = Q_PREFIX + text
    else:
        text = D_PREFIX + text

    vec = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    vec = vec.tolist()
    if EMB_DIM and len(vec) != EMB_DIM:
        raise RuntimeError(f"Embedding dim={len(vec)} != EMB_DIM={EMB_DIM} (check ES mapping & model)")
    return vec


def embed_query(meta: Dict[str, Any]) -> Optional[List[float]]:
    rewrite = meta.get("query_rewrite") or []
    norm = meta.get("keyword_normalized") or []
    should_kw = meta.get("should_keywords") or []

    parts: List[str] = []
    for x in rewrite:
        if isinstance(x, str) and x.strip():
            parts.append(x.strip())
    for x in norm:
        if isinstance(x, str) and x.strip():
            parts.append(x.strip())
    if not parts:
        for x in should_kw:
            if isinstance(x, str) and x.strip():
                parts.append(x.strip())

    text = " ; ".join(parts).strip()
    if not text:
        return None

    return embed_text(text, is_query=True)

def search_es(base_url: str, index: str, auth, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{index}/_search"
    r = requests.post(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        **request_kwargs(timeout=60, headers={"Content-Type": "application/json"})
    )
    if r.status_code >= 400:
        print("ES error:", r.status_code, r.text[:2000])
    r.raise_for_status()
    return r.json()

def build_knn_body(
    query_vector: List[float],
    size: int,
    meta: Dict[str, Any],
    reference_date: str
) -> Dict[str, Any]:
    effective_filters = build_effective_filter(meta, reference_date)

    knn_k = max(1, int(size))
    num_candidates = min(max(knn_k * 3, 100), 10000)

    return {
        "size": knn_k,
        "_source": [
            "law", "idd", "title", "title_dk", "title_law",
            "content_use", "content", "content_raw",
            "ngay_bh", "ngay_hl", "ngay_hhl",
            "link_online", "source_file"
        ],
        "knn": {
            "field": ES_VECTOR_FIELD,
            "query_vector": query_vector,
            "k": knn_k,
            "num_candidates": num_candidates,
            "filter": effective_filters
        }
    }

def knn_topk_effective(meta: Dict[str, Any], k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
    reference_date = get_reference_date(meta)
    auth = (ES_USER, ES_PASSWORD)

    fetch_size = max(k * 3, 100)

    qvec = embed_query(meta)
    if qvec is None:
        return reference_date, []

    knn_body = build_knn_body(qvec, fetch_size, meta, reference_date)
    knn_res = search_es(ES_BASE_URL, ES_INDEX, auth, knn_body)
    knn_hits = knn_res.get("hits", {}).get("hits", [])

    return reference_date, knn_hits[:k]