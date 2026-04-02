import os
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL_ID = os.getenv(
    "RERANK_MODEL_ID",
    "nguyenhoangtrungg/Tinix_Neco_Pretrained_ALQAC"
).strip()

RERANK_TOKENIZER_ID = os.getenv(
    "RERANK_TOKENIZER_ID",
    "bert-base-multilingual-cased"
).strip()

RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "384"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))

RERANK_POS_LABEL_IDX = int(os.getenv("RERANK_POS_LABEL_IDX", "1"))

RERANK_SINGLE_LOGIT_SIGMOID = os.getenv(
    "RERANK_SINGLE_LOGIT_SIGMOID", "true"
).lower() in ("1", "true", "yes", "y")

RERANK_DEBUG = os.getenv("RERANK_DEBUG", "false").lower() in ("1", "true", "yes", "y")
RERANK_DEBUG_MAX_SHOW = int(os.getenv("RERANK_DEBUG_MAX_SHOW", "5"))

RERANK_USE_FUSION_DEFAULT = os.getenv(
    "RERANK_USE_FUSION_DEFAULT", "true"
).lower() in ("1", "true", "yes", "y")
RERANK_ALPHA_RETRIEVAL = float(os.getenv("RERANK_ALPHA_RETRIEVAL", "0.7"))
RERANK_BETA_RERANK = float(os.getenv("RERANK_BETA_RERANK", "0.3"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Rerank device: {DEVICE}")
print(f"RERANK_MODEL_ID: {RERANK_MODEL_ID}")
print(f"RERANK_TOKENIZER_ID: {RERANK_TOKENIZER_ID}")
print(f"RERANK_MAX_LEN: {RERANK_MAX_LEN}")
print(f"RERANK_BATCH_SIZE: {RERANK_BATCH_SIZE}")
print(f"RERANK_POS_LABEL_IDX: {RERANK_POS_LABEL_IDX}")
print(f"RERANK_SINGLE_LOGIT_SIGMOID: {RERANK_SINGLE_LOGIT_SIGMOID}")
print(f"RERANK_USE_FUSION_DEFAULT: {RERANK_USE_FUSION_DEFAULT}")
print(f"RERANK_ALPHA_RETRIEVAL: {RERANK_ALPHA_RETRIEVAL}")
print(f"RERANK_BETA_RERANK: {RERANK_BETA_RERANK}")
print(f"RERANK_DEBUG: {RERANK_DEBUG}")

rerank_tokenizer = AutoTokenizer.from_pretrained(
    RERANK_TOKENIZER_ID,
    use_fast=True,
    trust_remote_code=True
)

rerank_model = AutoModelForSequenceClassification.from_pretrained(
    RERANK_MODEL_ID,
    trust_remote_code=True
)
rerank_model.to(DEVICE)
rerank_model.eval()

print("Loaded ALQAC reranker")
print(f"model_type : {getattr(rerank_model.config, 'model_type', None)}")
print(f"num_labels : {getattr(rerank_model.config, 'num_labels', None)}")
print(f"id2label   : {getattr(rerank_model.config, 'id2label', None)}")
print(f"label2id   : {getattr(rerank_model.config, 'label2id', None)}")

def _norm_text(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    return s


def build_doc_text_from_hit(hit: Dict[str, Any]):
    src = hit.get("_source", {}) or {}

    title = (
        _norm_text(src.get("title"))
        or _norm_text(src.get("title_dk"))
        or _norm_text(src.get("title_law"))
    )
    law = _norm_text(src.get("law"))
    content = (
        _norm_text(src.get("content_use"))
        or _norm_text(src.get("content"))
        or _norm_text(src.get("content_raw"))
    )

    parts = [title, law, content]
    parts = [p for p in parts if p]
    return "\n".join(parts).strip()


def _scores_from_logits(logits: torch.Tensor):
    logits = logits.detach().float().cpu()

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    num_labels = logits.shape[-1]

    if num_labels == 1:
        if RERANK_SINGLE_LOGIT_SIGMOID:
            return torch.sigmoid(logits[:, 0]).tolist()
        return logits[:, 0].tolist()

    probs = torch.softmax(logits, dim=-1)
    pos_idx = max(0, min(RERANK_POS_LABEL_IDX, num_labels - 1))
    return probs[:, pos_idx].tolist()


def _debug_logits(logits: torch.Tensor, start_idx: int = 0):
    if not RERANK_DEBUG:
        return

    x = logits.detach().float().cpu()
    if x.ndim == 1:
        x = x.unsqueeze(0)

    show_n = min(RERANK_DEBUG_MAX_SHOW, x.shape[0])
    print("\n===== ALQAC LOGITS DEBUG =====")
    print(f"logits.shape = {tuple(x.shape)}")

    if x.shape[-1] == 1:
        vals = x[:show_n, 0].tolist()
        sigs = torch.sigmoid(x[:show_n, 0]).tolist()
        for i in range(show_n):
            print(f"[{start_idx + i}] raw_logit={vals[i]:.6f} sigmoid={sigs[i]:.6f}")
    else:
        probs = torch.softmax(x[:show_n], dim=-1)
        for i in range(show_n):
            print(f"[{start_idx + i}] logits={x[i].tolist()} probs={probs[i].tolist()}")

    print("================================\n")


@torch.no_grad()
def score_pairs_alqac(
    pairs: List[Tuple[str, str]],
    batch_size: int = RERANK_BATCH_SIZE,
    max_length: int = RERANK_MAX_LEN
):
    all_scores: List[float] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]

        questions = [q for q, _ in batch]
        docs = [d for _, d in batch]

        enc = rerank_tokenizer(
            questions,
            docs,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        outputs = rerank_model(**enc)
        logits = outputs.logits

        _debug_logits(logits, start_idx=start)

        batch_scores = _scores_from_logits(logits)
        all_scores.extend(batch_scores)

    return all_scores


def _minmax_norm(vals: List[float]):
    if not vals:
        return []
    mn = min(vals)
    mx = max(vals)
    if abs(mx - mn) < 1e-12:
        return [0.0 for _ in vals]
    return [(x - mn) / (mx - mn) for x in vals]

def rerank_with_alqac(
    query: str,
    meta: Dict[str, Any],
    hits: List[Dict[str, Any]],
    top_n: int = 10,
    use_score_fusion: bool = RERANK_USE_FUSION_DEFAULT,
    alpha_retrieval: float = RERANK_ALPHA_RETRIEVAL,
    beta_rerank: float = RERANK_BETA_RERANK,
):
    if not hits:
        return {"top_keys": [], "top_hits": [], "scores": []}

    pairs: List[Tuple[str, str]] = []
    kept_hits: List[Dict[str, Any]] = []

    for h in hits:
        doc_text = build_doc_text_from_hit(h)
        if not doc_text:
            continue
        pairs.append((query, doc_text))
        kept_hits.append(h)

    if not kept_hits:
        return {"top_keys": [], "top_hits": [], "scores": []}

    rerank_scores = score_pairs_alqac(pairs)

    for h, s in zip(kept_hits, rerank_scores):
        h["_alqac_score"] = float(s)

    if use_score_fusion:
        retrieval_scores = [float(h.get("_score", 0.0)) for h in kept_hits]
        retrieval_norm = _minmax_norm(retrieval_scores)
        rerank_norm = _minmax_norm(rerank_scores)

        for h, rs, zs in zip(kept_hits, retrieval_norm, rerank_norm):
            h["_final_score"] = alpha_retrieval * rs + beta_rerank * zs

        kept_hits.sort(key=lambda x: x["_final_score"], reverse=True)
        final_scores = [float(h["_final_score"]) for h in kept_hits[:top_n]]
    else:
        kept_hits.sort(key=lambda x: x["_alqac_score"], reverse=True)
        final_scores = [float(h["_alqac_score"]) for h in kept_hits[:top_n]]

    top_hits = kept_hits[:top_n]
    top_keys = []

    for h in top_hits:
        src = h.get("_source", {}) or {}
        key = str(src.get("idd") or h.get("_id") or "").strip()
        if key:
            top_keys.append(key)

    return {
        "top_keys": top_keys,
        "top_hits": top_hits,
        "scores": final_scores,
    }


def debug_print_rerank_output(out: Dict[str, Any], max_show: int = 10):
    top_hits = out.get("top_hits", [])[:max_show]
    print("\n===== ALQAC RERANK OUTPUT =====")
    for i, h in enumerate(top_hits, 1):
        src = h.get("_source", {}) or {}
        print(f"[{i}] idd={src.get('idd')}")
        print(f"    law         : {src.get('law')}")
        print(f"    title       : {src.get('title') or src.get('title_dk') or src.get('title_law')}")
        print(f"    retrieval   : {h.get('_score')}")
        print(f"    alqac_score : {h.get('_alqac_score')}")
        print(f"    final_score : {h.get('_final_score')}")
        print()