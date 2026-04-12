import os
import sys
import json
import requests
import urllib3
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from utils.es_http import request_kwargs
from utils.effective_filter import get_reference_date, build_effective_filter, DEFAULT_WEIGHTS

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

BASE_MODEL = os.getenv("HF_MODEL", "jinaai/jina-embeddings-v3")
MAX_TOKENS = int(os.getenv("EMB_MAX_TOKENS", "512"))
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device: {device} | dtype: {dtype}")
print(f"BASE_MODEL: {BASE_MODEL}")

def _align_cos_sin(x: torch.Tensor, cos, sin):
    if not torch.is_tensor(cos):
        cos = torch.tensor(cos, device=x.device, dtype=x.dtype)
    else:
        cos = cos.to(device=x.device, dtype=x.dtype)

    if not torch.is_tensor(sin):
        sin = torch.tensor(sin, device=x.device, dtype=x.dtype)
    else:
        sin = sin.to(device=x.device, dtype=x.dtype)

    if cos.ndim == 2 and x.ndim == 4:
        S = cos.shape[0]

        if S == x.shape[1]:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)

        elif S == x.shape[2]:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            Smin = min(S, x.shape[1])
            cos = cos[:Smin].unsqueeze(0).unsqueeze(2)
            sin = sin[:Smin].unsqueeze(0).unsqueeze(2)

    return cos, sin

def _apply_rotary_torch(x: torch.Tensor, cos, sin, *args, **kwargs):
    cos, sin = _align_cos_sin(x, cos, sin)

    x_dim = x.shape[-1]
    rot_dim = cos.shape[-1]

    if rot_dim * 2 <= x_dim:
        x1 = x[..., :rot_dim]
        x2 = x[..., rot_dim:rot_dim * 2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        if rot_dim * 2 < x_dim:
            rest = x[..., rot_dim * 2:]
            return torch.cat([out1, out2, rest], dim=-1)
        return torch.cat([out1, out2], dim=-1)

    raise RuntimeError(
        f"apply_rotary patch unsupported shapes: x={tuple(x.shape)} cos={tuple(cos.shape)}"
    )

def _patch_jina_rotary():
    patched = False
    for name, mod in list(sys.modules.items()):
        if name.endswith(".rotary") and ("xlm" in name or "jina" in name):
            setattr(mod, "apply_rotary", _apply_rotary_torch)
            print("Patched apply_rotary in:", name)
            patched = True
            break
    if not patched:
        print("rotary module not found yet (will patch again after model load).")

_patch_jina_rotary()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
model = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)
model.to(device)
model.eval()

_patch_jina_rotary()

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

@torch.no_grad()
def embed_text(text: str) -> Optional[List[float]]:
    text = (text or "").strip()
    if not text:
        return None
    
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and tmax > 0:
        max_len = min(MAX_TOKENS, tmax)
    else:
        max_len = MAX_TOKENS

    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = F.normalize(emb, p=2, dim=-1)

    return emb[0].detach().float().cpu().tolist()

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

    v = embed_text(text)
    if v is None:
        return None

    if EMB_DIM and len(v) != EMB_DIM:
        raise RuntimeError(f"Query embedding dim={len(v)} != EMB_DIM={EMB_DIM}")

    return v

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
    return {
        "size": size,
        "_source": [
            "law", "idd", "title", "title_dk", "title_law",
            "content_use", "content", "content_raw",
            "ngay_bh", "ngay_hl", "ngay_hhl",
            "link_online", "source_file"
        ],
        "knn": {
            "field": ES_VECTOR_FIELD,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size * 5, 50),
            "filter": effective_filters
        }
    }


def knn_topk_effective(meta: Dict[str, Any], k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
    reference_date = get_reference_date(meta)
    auth = (ES_USER, ES_PASSWORD)

    fetch_size = max(k * 8, 80)

    qvec = embed_query(meta)
    if qvec is None:
        return reference_date, []

    knn_body = build_knn_body(qvec, fetch_size, meta, reference_date)
    knn_res = search_es(ES_BASE_URL, ES_INDEX, auth, knn_body)
    knn_hits = knn_res.get("hits", {}).get("hits", [])

    return reference_date, knn_hits[:k]