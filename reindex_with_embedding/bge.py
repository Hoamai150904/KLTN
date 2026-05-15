import os
import json
import time
import requests
import urllib3
from dotenv import load_dotenv
from utils.es_http import request_kwargs
from typing import List, Dict, Any

import torch
from sentence_transformers import SentenceTransformer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()


ES_URL = (os.getenv("ES_BASE_URL") or os.getenv("ES_URL") or "https://localhost:9200").rstrip("/")
SRC_INDEX = os.getenv("SRC_INDEX", "cu_tru_law_v4")
DST_INDEX = os.getenv("DST_INDEX", "cu_tru_law_v4_bge")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_API_KEY = os.getenv("ES_API_KEY", "").strip()
AUTH = (ES_USER, ES_PASSWORD) if (ES_USER and not ES_API_KEY) else None
VERIFY_TLS = os.getenv("ES_VERIFY_TLS", "false").lower() in ("1", "true", "yes", "y")

VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "content_vector")

BATCH = int(os.getenv("BATCH", "200"))
BULK_BATCH = int(os.getenv("BULK_BATCH", "50")) 

BASE_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
EMB_MAX_LEN = int(os.getenv("EMB_MAX_LEN", "256"))
EMB_BATCH = int(os.getenv("EMB_BATCH", "16"))
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))

D_PREFIX = os.getenv("D_PREFIX", "Represent this document for retrieval: ")

HF_TOKEN = os.getenv("HF_TOKEN", None)

CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))

def load_embedder():
    if not torch.cuda.is_available() and CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)

    print("BASE_MODEL:", BASE_MODEL)
    print("CUDA available:", torch.cuda.is_available())

    if HF_TOKEN:
        m = SentenceTransformer(BASE_MODEL, token=HF_TOKEN)
    else:
        m = SentenceTransformer(BASE_MODEL)

    m.max_seq_length = EMB_MAX_LEN
    print("max_seq_length:", m.max_seq_length)
    return m

EMB_MODEL_OBJ = load_embedder()

@torch.inference_mode()
def embed_texts(texts: List[str]) -> List[List[float]]:
    batch = []
    for t in texts:
        t = (t or "").strip()
        batch.append(D_PREFIX + t if t else D_PREFIX)

    vecs = EMB_MODEL_OBJ.encode(
        batch,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=min(EMB_BATCH, 64),
    )

    vecs = vecs.tolist()

    if EMB_DIM:
        for v in vecs:
            if len(v) != EMB_DIM:
                raise RuntimeError(f"Embedding dim={len(v)} != EMB_DIM={EMB_DIM} (check model or env)")

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

        vecs = embed_texts(pending_texts)

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
        print(f"\nReindex + embedding (HF SentenceTransformer) DONE. Total indexed: {total} | time={elapsed:.1f}s")

    finally:
        close_pit(pit_id)
        print("PIT closed")

if __name__ == "__main__":
    main()