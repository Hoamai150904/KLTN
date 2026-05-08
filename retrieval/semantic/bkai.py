import os
import json
import requests
import urllib3
from typing import Dict, Any, List, Tuple, Optional

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

HF_MODEL = os.getenv("HF_MODEL", "bkai-foundation-models/vietnamese-bi-encoder")
MAX_TOKENS = int(os.getenv("EMB_MAX_TOKENS", "256"))
EMB_DIM_ENV = os.getenv("EMB_DIM", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", None)

ENABLE_WORD_SEGMENT = os.getenv("ENABLE_WORD_SEGMENT", "true").lower() in ("1", "true", "yes", "y")

_SEGMENTER_NAME = None
ViTokenizer = None
underthesea_word_tokenize = None

if ENABLE_WORD_SEGMENT:
    try:
        from pyvi import ViTokenizer
        _SEGMENTER_NAME = "pyvi"
        print("Vietnamese segmenter: pyvi")
    except Exception:
        try:
            from underthesea import word_tokenize as underthesea_word_tokenize
            _SEGMENTER_NAME = "underthesea"
            print("Vietnamese segmenter: underthesea")
        except Exception:
            _SEGMENTER_NAME = None
            print("No Vietnamese segmenter found. Query text will be used as-is.")

print(f"HF_MODEL: {HF_MODEL}")

if HF_TOKEN:
    st_model = SentenceTransformer(HF_MODEL, token=HF_TOKEN)
else:
    st_model = SentenceTransformer(HF_MODEL)

try:
    st_model.max_seq_length = MAX_TOKENS
except Exception:
    pass

EMB_DIM = int(EMB_DIM_ENV) if EMB_DIM_ENV.isdigit() else int(st_model.get_sentence_embedding_dimension())
print(f"ST max_seq_length: {getattr(st_model, 'max_seq_length', None)} | EMB_DIM: {EMB_DIM}")

def segment_vietnamese_text(text: str) -> str:
    text = (text or "").strip()
    if not text or not ENABLE_WORD_SEGMENT or not _SEGMENTER_NAME:
        return text

    try:
        if _SEGMENTER_NAME == "pyvi" and ViTokenizer is not None:
            return ViTokenizer.tokenize(text)

        if _SEGMENTER_NAME == "underthesea" and underthesea_word_tokenize is not None:
            out = underthesea_word_tokenize(text, format="text")
            if isinstance(out, list):
                return " ".join(out)
            return str(out)
    except Exception:
        pass

    return text


def prepare_text_for_embedding(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = segment_vietnamese_text(text)
    return text.strip()

def embed_text(text: str) -> Optional[List[float]]:
    text = prepare_text_for_embedding(text)
    if not text:
        return None

    vec = st_model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    v = vec.tolist()
    if EMB_DIM and len(v) != EMB_DIM:
        raise RuntimeError(f"Query embedding dim={len(v)} != EMB_DIM={EMB_DIM}")

    return v


def pick_best_query_text(meta: Dict[str, Any]) -> str:
    rewrite = meta.get("query_rewrite") or []
    original = meta.get("keywords_original") or []
    should_kw = meta.get("should_keywords") or []
    norm = meta.get("keyword_normalized") or []

    for x in rewrite:
        if isinstance(x, str) and x.strip():
            return x.strip()

    for x in original:
        if isinstance(x, str) and x.strip():
            return x.strip()

    for x in should_kw:
        if isinstance(x, str) and x.strip():
            return x.strip()

    parts: List[str] = []
    for x in norm:
        if isinstance(x, str) and x.strip():
            parts.append(x.strip())

    return " ".join(parts).strip()


def embed_query(meta: Dict[str, Any]) -> Optional[List[float]]:
    text = pick_best_query_text(meta)
    if not text:
        return None
    return embed_text(text)


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