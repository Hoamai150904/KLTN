import os
import json
import requests
import urllib3
from typing import Dict, Any, List, Tuple, Optional
from datetime import date, datetime
from utils.effective_filter import build_effective_filter, get_reference_date, DEFAULT_WEIGHTS

from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

ES_BASE_URL = (os.getenv("TFIDF_ES_URL") or "http://localhost:9200").rstrip("/")
ES_INDEX = os.getenv("TFIDF_ES_INDEX", "cu_tru_law_v4")


def _build_query_text(meta: Dict[str, Any]) -> str:
    rewrite = meta.get("query_rewrite") or []
    norm = meta.get("keyword_normalized") or []
    should_kw = meta.get("should_keywords") or []

    parts: List[str] = []
    for x in (rewrite + norm + should_kw):
        if isinstance(x, str) and x.strip():
            parts.append(x.strip())

    parts = list(dict.fromkeys(parts))
    return " ".join(parts).strip()


def build_tfidf_query_from_meta(meta: Dict[str, Any], size: int = 10) -> Dict[str, Any]:
    boosts = dict(DEFAULT_WEIGHTS)
    boosts.update(meta.get("boosts") or {})
    prefer_version = (meta.get("prefer_version") or "both").lower()

    b_title = float(boosts.get("title", 2.2))
    b_title_dk = float(boosts.get("title_dk", 2.0))
    b_title_law = float(boosts.get("title_law", 1.8))
    b_content_use = float(boosts.get("content_use", 1.3))
    b_content = float(boosts.get("content", 1.0))
    b_content_raw = float(boosts.get("content_raw", 0.8))
    b_idd = float(boosts.get("idd", 0.6))
    b_law = float(boosts.get("law", 0.6))
    b_idsdbs = float(boosts.get("idsdbs", 0.2))
    b_link_online = float(boosts.get("link_online", 0.1))
    b_source_file = float(boosts.get("source_file", 0.1))

    if prefer_version == "merged":
        b_content_use *= 1.10
        b_content *= 0.90
    elif prefer_version == "original":
        b_content *= 1.10
        b_content_use *= 0.90

    query_text = _build_query_text(meta)
    reference_date = get_reference_date(meta)
    effective_filters = build_effective_filter(meta, reference_date)

    fields = [
        f"title^{b_title}",
        f"title_dk^{b_title_dk}",
        f"title_law^{b_title_law}",
        f"content_use^{b_content_use}",
        f"content^{b_content}",
        f"content_raw^{b_content_raw}",
        f"idd^{b_idd}",
        f"law^{b_law}",
        f"idsdbs^{b_idsdbs}",
        f"link_online^{b_link_online}",
        f"source_file^{b_source_file}",
    ]

    should_clauses: List[Dict[str, Any]] = []
    if query_text:
        should_clauses.append({
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "best_fields",
                "operator": "or",
                "tie_breaker": 0.2,
                "minimum_should_match": "30%"
            }
        })

        should_clauses.append({
            "match_phrase": {
                "title": {"query": query_text, "boost": 3.0}
            }
        })
        should_clauses.append({
            "match_phrase": {
                "title_dk": {"query": query_text, "boost": 2.7}
            }
        })
        should_clauses.append({
            "match_phrase": {
                "title_law": {"query": query_text, "boost": 2.3}
            }
        })

        should_clauses.append({
            "match_phrase": {
                "title_dk": {"query": query_text, "slop": 2, "boost": 1.8}
            }
        })
        should_clauses.append({
            "match_phrase": {
                "title_law": {"query": query_text, "slop": 2, "boost": 1.5}
            }
        })

    query = {
        "bool": {
            "filter": effective_filters,
            "should": should_clauses if query_text else [{"match_all": {}}],
            "minimum_should_match": 1
        }
    }

    return {
        "size": size,
        "fields": [
            "law", "idd", "title", "title_dk", "title_law",
            "content_use", "content", "content_raw",
            "ngay_bh", "ngay_hl", "ngay_hhl",
            "link_online", "source_file", "idsdbs"
        ],
        "query": query
    }


def search_es(base_url: str, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{index}/_search"
    r = requests.post(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if r.status_code >= 400:
        print("ES error:", r.status_code, r.text[:2000])
    r.raise_for_status()
    return r.json()


def tfidf_topk_effective(meta: Dict[str, Any], k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
    reference_date = get_reference_date(meta)

    fetch_size = max(k * 8, 80)

    tfidf_body = build_tfidf_query_from_meta(meta, size=fetch_size)
    tfidf_res = search_es(ES_BASE_URL, ES_INDEX, tfidf_body)

    raw_hits = tfidf_res.get("hits", {}).get("hits", [])
    hits = []
    for h in raw_hits:
        fields = h.get("fields", {})
        src_like = {}
        for kf, vf in fields.items():
            if isinstance(vf, list) and len(vf) == 1:
                src_like[kf] = vf[0]
            else:
                src_like[kf] = vf
        hits.append({
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "_source": src_like
        })

    return reference_date, hits[:k]