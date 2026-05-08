import copy
from typing import Dict, Any, List, Tuple, Optional

from retrieval.keyword.bm25 import bm25_topk_effective
from retrieval.semantic.bkai import knn_topk_effective

EPS = 1e-12


def _safe_score(hit: Dict[str, Any]) -> float:
    try:
        return float(hit.get("_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _doc_key(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {}) or {}

    for key in ("idd", "idsdbs"):
        v = src.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()

    _id = hit.get("_id")
    if _id is not None and str(_id).strip():
        return str(_id).strip()

    return str(id(hit))


def _minmax_normalize(hits: List[Dict[str, Any]]) -> Dict[str, float]:
    if not hits:
        return {}

    pairs = [(_doc_key(h), _safe_score(h)) for h in hits]
    scores = [s for _, s in pairs]

    s_min = min(scores)
    s_max = max(scores)

    if abs(s_max - s_min) <= EPS:
        return {doc_id: 1.0 for doc_id, _ in pairs}

    return {
        doc_id: (score - s_min) / (s_max - s_min + EPS)
        for doc_id, score in pairs
    }


def _merge_hybrid(
    bm25_hits: List[Dict[str, Any]],
    bkai_hits: List[Dict[str, Any]],
    alpha: float
) -> List[Dict[str, Any]]:
    bm25_norm = _minmax_normalize(bm25_hits)
    bkai_norm = _minmax_normalize(bkai_hits)

    merged: Dict[str, Dict[str, Any]] = {}

    for hit in bkai_hits:
        merged[_doc_key(hit)] = copy.deepcopy(hit)

    for hit in bm25_hits:
        doc_id = _doc_key(hit)
        if doc_id not in merged:
            merged[doc_id] = copy.deepcopy(hit)

    results: List[Dict[str, Any]] = []
    for doc_id, hit in merged.items():
        b = bm25_norm.get(doc_id, 0.0)
        s = bkai_norm.get(doc_id, 0.0)
        hybrid_score = alpha * b + (1.0 - alpha) * s

        hit["_score"] = hybrid_score
        results.append(hit)

    results.sort(key=lambda x: float(x.get("_score", 0.0)), reverse=True)
    return results


def hybrid_topk_effective(
    meta: Dict[str, Any],
    k: int = 10,
    alpha: float = 0.1,
    bm25_fetch_k: Optional[int] = None,
    bkai_fetch_k: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha phải nằm trong [0, 1]")

    if bm25_fetch_k is None:
        bm25_fetch_k = max(k, 150)
    if bkai_fetch_k is None:
        bkai_fetch_k = max(k, 150)

    ref_bm25, bm25_hits = bm25_topk_effective(meta, k=bm25_fetch_k)
    ref_bkai, bkai_hits = knn_topk_effective(meta, k=bkai_fetch_k)

    reference_date = ref_bkai or ref_bm25

    if not bm25_hits and not bkai_hits:
        return reference_date, []

    if not bm25_hits:
        only_bkai = copy.deepcopy(bkai_hits[:k])
        bkai_norm = _minmax_normalize(bkai_hits)
        for hit in only_bkai:
            hit["_score"] = bkai_norm.get(_doc_key(hit), 0.0)
        return reference_date, only_bkai

    if not bkai_hits:
        only_bm25 = copy.deepcopy(bm25_hits[:k])
        bm25_norm = _minmax_normalize(bm25_hits)
        for hit in only_bm25:
            hit["_score"] = bm25_norm.get(_doc_key(hit), 0.0)
        return reference_date, only_bm25

    merged_hits = _merge_hybrid(bm25_hits, bkai_hits, alpha=alpha)
    return reference_date, merged_hits[:k]