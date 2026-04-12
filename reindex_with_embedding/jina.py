import os
import json
import time
import requests
import urllib3
from dotenv import load_dotenv
from utils.es_http import request_kwargs
from typing import List, Dict, Any, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

ES_URL = (os.getenv("ES_BASE_URL") or os.getenv("ES_URL") or "https://localhost:9200").rstrip("/")
SRC_INDEX = os.getenv("SRC_INDEX", "cu_tru_law_v4")
DST_INDEX = os.getenv("DST_INDEX", "cu_tru_law_v4_jina_emb")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
AUTH = (ES_USER, ES_PASSWORD) if (ES_USER and not ES_API_KEY) else None
VERIFY_TLS = os.getenv("ES_VERIFY_TLS", "false").lower() in ("1", "true", "yes", "y")

VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "content_vector")

BATCH = 200
BULK_BATCH = 50 

BASE_MODEL = os.getenv("EMB_MODEL", "jinaai/jina-embeddings-v3")
EMB_MAX_LEN = int(os.getenv("EMB_MAX_LEN", "512"))  
EMB_BATCH = int(os.getenv("EMB_BATCH", "16"))
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))

CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if device == "cpu" and CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)

    print(f"Embedding device: {device} | dtype: {dtype}")
    print(f"BASE_MODEL: {BASE_MODEL}")

    _patch_jina_rotary()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    model.to(device)

    _patch_jina_rotary()

    return tok, model, device


TOKENIZER, EMB_MODEL_OBJ, EMB_DEVICE = load_embedder()


@torch.inference_mode()
def embed_texts(texts: List[str], max_length: int) -> List[List[float]]:
    texts = [(t or "").strip() for t in texts]
    enc = TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    enc = {k: v.to(EMB_DEVICE) for k, v in enc.items()}

    out = EMB_MODEL_OBJ(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = F.normalize(emb, p=2, dim=-1)

    vecs = emb.detach().float().cpu().tolist()

    if EMB_DIM:
        for v in vecs:
            if len(v) != EMB_DIM:
                raise RuntimeError(f"Embedding dim={len(v)} != EMB_DIM={EMB_DIM}")

    return vecs

def open_pit(keep_alive="100m") -> str:
    url = f"{ES_URL}/{SRC_INDEX}/_pit?keep_alive={keep_alive}"
    r = requests.post(url, **request_kwargs(timeout=60))
    r.raise_for_status()
    return r.json()["id"]


def close_pit(pit_id: str):
    url = f"{ES_URL}/_pit"
    r = requests.delete(url, json={"id": pit_id}, **request_kwargs(timeout=60))
    if r.status_code not in (200, 404):
        r.raise_for_status()


def search_page(pit_id: str, search_after=None) -> dict:
    url = f"{ES_URL}/_search"
    body = {
        "size": BATCH,
        "pit": {"id": pit_id, "keep_alive": "5m"},
        "sort": [{"_shard_doc": "asc"}],
        "_source": True,
    }
    if search_after is not None:
        body["search_after"] = search_after

    r = requests.post(url,  json=body, **request_kwargs(timeout=60))
    r.raise_for_status()
    return r.json()


def bulk_index(lines: List[str]):
    url = f"{ES_URL}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    payload = "\n".join(lines) + "\n"
    r = requests.post(
        url,
        data=payload.encode("utf-8"),
        **request_kwargs(timeout=180, headers=headers)
    )
    r.raise_for_status()
    out = r.json()
    if out.get("errors"):
        for item in out.get("items", [])[:10]:
            act = item.get("index") or item.get("create") or item.get("update") or {}
            if "error" in act:
                print("Bulk item error:", json.dumps(act["error"], ensure_ascii=False, indent=2))
                break
        raise RuntimeError("Bulk indexing has errors.")
    return out

def main():
    pit_id = open_pit()
    print("PIT opened")

    total = 0
    search_after = None
    buf: List[str] = []

    pending_ids: List[str] = []
    pending_docs: List[Dict[str, Any]] = []
    pending_texts: List[str] = []

    t0 = time.time()

    def flush_embedding_batch():
        nonlocal total, buf, pending_ids, pending_docs, pending_texts

        if not pending_texts:
            return

        vecs = embed_texts(pending_texts, EMB_MAX_LEN)

        for doc_id, src, vec in zip(pending_ids, pending_docs, vecs):
            src[VECTOR_FIELD] = vec
            buf.append(json.dumps({"index": {"_index": DST_INDEX, "_id": doc_id}}, ensure_ascii=False))
            buf.append(json.dumps(src, ensure_ascii=False))

            if len(buf) >= BULK_BATCH * 2:
                bulk_index(buf)
                total += BULK_BATCH
                elapsed = time.time() - t0
                print(f"Indexed ~{total} docs... ({elapsed:.1f}s)", end="\r")
                buf = []

        pending_ids, pending_docs, pending_texts = [], [], []

    try:
        while True:
            res = search_page(pit_id, search_after=search_after)
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                break

            for h in hits:
                doc_id = h["_id"]
                src = h.get("_source") or {}

                text = src.get("content_use") or src.get("content") or src.get("content_raw")
                if not text:
                    continue

                pending_ids.append(doc_id)
                pending_docs.append(src)
                pending_texts.append(text)

                if len(pending_texts) >= EMB_BATCH:
                    flush_embedding_batch()

            search_after = hits[-1]["sort"]

        flush_embedding_batch()

        if buf:
            bulk_index(buf)
            total += len(buf) // 2
            buf = []

        elapsed = time.time() - t0
        print(f"\nReindex + embedding (Jina) DONE. Total indexed: {total} | time={elapsed:.1f}s")

    finally:
        close_pit(pit_id)
        print("PIT closed")


if __name__ == "__main__":
    main()
