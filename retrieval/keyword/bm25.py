import os
import json
import requests
import urllib3
from typing import Dict, Any
from utils.effective_filter import get_reference_date, build_effective_filter, DEFAULT_WEIGHTS

from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

ES_BASE_URL = os.getenv("ES_BASE_URL")
ES_INDEX = os.getenv("ES_INDEX") 
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
AUTH = (ES_USER, ES_PASSWORD)

VERIFY_TLS = os.getenv("ES_VERIFY_TLS", "false").lower() in ("1", "true", "yes", "y")

def build_bm25_query_from_meta(meta: Dict[str, Any], size: int = 10):
    boosts = meta.get("boosts") or DEFAULT_WEIGHTS
    prefer_version = (meta.get("prefer_version") or "both").lower()

    b_title_dk = boosts.get("title_dk", 0.95)
    b_title_law = boosts.get("title_law", 0.95)
    b_content_use = boosts.get("content_use", 0.7)
    b_content = boosts.get("content", 0.7)
    b_content_raw = boosts.get("content_raw", 0.7)

    if prefer_version == "merged":
        b_content_use *= 1.8
        b_content *= 0.9
    elif prefer_version == "original":
        b_content *= 1.8
        b_content_use *= 0.9

    fields = [
        f"title_dk^{b_title_dk}",
        f"title_law^{b_title_law}",
        f"content_use^{b_content_use}",
        f"content^{b_content}",
        f"content_raw^{b_content_raw}",
        f"idd^{boosts.get('idd', 0.3)}",
        f"law^{boosts.get('law', 0.3)}",
        f"idsdbs^{boosts.get('idsdbs', 0.2)}",
        f"link_online^{boosts.get('link_online', 0.1)}",
        f"source_file^{boosts.get('source_file', 0.1)}",
    ]

    should_list = meta.get("should_keywords") or []
    norm = meta.get("keyword_normalized") or []
    rewrite = meta.get("query_rewrite") or []
    original = meta.get("keywords_original") or []

    should_combo = list(dict.fromkeys(rewrite + norm + should_list + original))
    should_q = " ".join(should_combo).strip()

    reference_date = get_reference_date(meta)
    effective_filters = build_effective_filter(meta, reference_date)

    query = {
        "bool": {
            "filter": effective_filters,
            "must": [{
                "multi_match": {
                    "query": should_q,
                    "fields": fields,
                    "type": "best_fields",
                    "operator": "or"
                }
            }] if should_q else [{"match_all": {}}]
        }
    }

    return {
        "size": size,
        "track_scores": True,
        "_source": [
            "law", "idd", "title", "title_dk", "title_law",
            "content_use", "content", "content_raw",
            "ngay_bh", "ngay_hl", "ngay_hhl",
            "link_online", "source_file"
        ],
        "query": query
    }

def search_es(base_url: str, index: str, auth, body: Dict[str, Any]):
    url = f"{base_url.rstrip('/')}/{index}/_search"
    r = requests.post(
        url,
        auth=auth,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        verify=VERIFY_TLS,
        timeout=60
    )
    if r.status_code >= 400:
        print("ES error:", r.status_code, r.text[:2000])
    r.raise_for_status()
    return r.json()

def bm25_topk_effective(meta: Dict[str, Any], k: int = 10):
    reference_date = get_reference_date(meta)
    auth = (ES_USER, ES_PASSWORD)

    fetch_size = max(k * 8, 80)

    bm25_body = build_bm25_query_from_meta(meta, size=fetch_size)
    bm25_res = search_es(ES_BASE_URL, ES_INDEX, auth, bm25_body)
    bm25_hits = bm25_res.get("hits", {}).get("hits", [])

    return reference_date, bm25_hits[:k]