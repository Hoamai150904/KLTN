import os
import re
from typing import List, Set, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from extract_metadata.extract_metadata import extract_metadata
from retrieval.keyword.bm25 import bm25_topk_effective
# from retrieval.semantic.jina import knn_topk_effective
from rerank.alqac import rerank_with_alqac

load_dotenv()

EVAL_PATH = os.getenv("EVAL_PATH", "data/groundtruthdatahc.xlsx")
SHEET_NAME = os.getenv("EVAL_SHEET", None)

K_LIST = [1, 2, 3, 5, 10, 20, 30, 50, 100]
FETCH_K = int(os.getenv("EVAL_FETCH_K", "150"))

COL_QUERY = os.getenv("EVAL_COL_QUERY", "question_note")
COL_DATE  = os.getenv("EVAL_COL_DATE", "post_date")
COL_GT    = os.getenv("EVAL_COL_GT", "idd_laws")

OUT_PATH = os.getenv("EVAL_OUT_LOCAL", "results.xlsx")
FORCE_EFFECTIVE = os.getenv("EVAL_FORCE_EFFECTIVE", "true").lower() in ("1", "true", "yes", "y")


def parse_idd_laws(cell):
    if cell is None:
        return set()
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return set()
    return {p.strip() for p in s.split(",") if p.strip()}


def safe_iso_date(x):
    from datetime import date, datetime
    if x is None:
        return date.today().isoformat()
    s = str(x).strip()
    if not s or s.lower() in ("nan", "nat", "none", "null"):
        return date.today().isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return date.today().isoformat()


def recall_at_k(pred: List[str], rel: Set[str]):
    if not rel:
        return 0.0
    return len(set(pred) & rel) / len(rel)

def precision_at_k(pred: List[str], rel: Set[str]):
    if not pred:
        return 0.0
    return len(set(pred) & rel) / len(pred)


def f1_at_k(pred: List[str], rel: Set[str]):
    p = precision_at_k(pred, rel)
    r = recall_at_k(pred, rel)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(pred: List[str], rel: Set[str]):
    for i, p in enumerate(pred, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def average_precision(pred: List[str], rel: Set[str]):
    if not rel:
        return 0.0
    hit = 0
    score = 0.0
    for i, p in enumerate(pred, start=1):
        if p in rel:
            hit += 1
            score += hit / i
    return score / len(rel)


def main():
    if EVAL_PATH.lower().endswith(".csv"):
        df = pd.read_csv(EVAL_PATH, encoding="utf-8-sig")
    else:
        xls = pd.ExcelFile(EVAL_PATH)
        df = pd.read_excel(EVAL_PATH, sheet_name=SHEET_NAME or xls.sheet_names[0])

    df.columns = [c.strip() for c in df.columns]

    writer = pd.ExcelWriter(OUT_PATH, engine="xlsxwriter")

    MAX_K = max(K_LIST)
    CANDIDATE_K = max(FETCH_K, MAX_K)

    summary_rows: List[Dict[str, Any]] = []

    detail_rows = []

    n_total = 0
    n_valid = 0
    n_meta_fail = 0
    n_retrieve_fail = 0
    n_rerank_fail = 0
    shown_meta = shown_ret = shown_rer = 0

    rr_sum = {k: 0.0 for k in K_LIST}
    rec_sum = {k: 0.0 for k in K_LIST}
    ap_sum = {k: 0.0 for k in K_LIST}
    prec_sum = {k: 0.0 for k in K_LIST}
    f1_sum = {k: 0.0 for k in K_LIST}

    for idx, r in df.iterrows():
        # Step 1: query
        query = str(r.get(COL_QUERY, "")).strip()
        if not query:
            continue

        n_total += 1
        rel_set = parse_idd_laws(r.get(COL_GT))
        reference_date = safe_iso_date(r.get(COL_DATE))

        # Step 2: extract metadata
        # try:
        #     meta = extract_metadata(query)
        # except Exception as e:
        #     n_meta_fail += 1
        #     if shown_meta < 3:
        #         print(f"[META FAIL] row={idx} err={e}")
        #         shown_meta += 1
        #     continue

        # meta["date_hint"] = reference_date
        # if FORCE_EFFECTIVE:
        #     meta["effective_status"] = "con_hieu_luc"
        # else:
        #     if meta.get("effective_status") in (None, "unknown"):
        #         meta["effective_status"] = "con_hieu_luc"

        meta = {
            "boosts": {
                "title_dk": 0.95, "title_law": 0.95,
                "content_use": 0.7, "content": 0.7, "content_raw": 0.7,
                "idd": 0.3, "law": 0.3, "idsdbs": 0.2,
                "source_file": 0.1, "link_online": 0.1,
                "ngay_bh": 0.0, "ngay_hl": 0.0, "ngay_hhl": 0.0,
            },

            "must_keywords": [],
            "should_keywords": [query],
            "exclude_keywords": [],
            "law_numbers": [],

            "prefer_version": "both",
            "effective_status": "con_hieu_luc",
            "date_hint": reference_date,

            "keyword_normalized": [],
            "query_rewrite": [],
            "keywords_original": [query],

            "intent": "unknown",
            "topic_candidates": [],
            "entities": {"actor": [], "relationship": [], "scenario": [], "location": ["unknown"]},
        }

        # Step 3: retrieval
        try:
            _, hits = bm25_topk_effective(meta, k=CANDIDATE_K)
            # _, hits = tfidf_topk_effective(meta, k=CANDIDATE_K, fetch_size=CANDIDATE_K)
            # _, hits = knn_topk_effective(meta, k=CANDIDATE_K)
            # _, hits = hybrid_tfidf_knn_effective(meta, k=CANDIDATE_K, fetch_size=CANDIDATE_K)
        except Exception as e:
            n_retrieve_fail += 1
            if shown_ret < 3:
                print(f"[RETRIEVE FAIL] row={idx} err={e}")
                shown_ret += 1
            continue

        # Step 4: rerank
        try:
            # rerank_out = rerank_with_bge(query, meta, hits, top_n=MAX_K)
            rerank_out = rerank_with_alqac(
                query=query,
                meta=meta,
                hits=hits,
                top_n=MAX_K
            )
            pred_full = [str(x) for x in rerank_out.get("top_keys", [])]
        except Exception as e:
            n_rerank_fail += 1
            if shown_rer < 3:
                print(f"[RERANK FAIL] row={idx} err={e}")
                shown_rer += 1
            pred_full = [str(h.get("_source", {}).get("idd") or h.get("_id")) for h in hits]

        pred_full = [p for p in pred_full if p and p.lower() not in ("none", "nan")]
        if len(pred_full) < MAX_K:
            hit_ids = [str(h.get("_source", {}).get("idd") or h.get("_id")) for h in hits]
            for hid in hit_ids:
                if hid and hid not in pred_full:
                    pred_full.append(hid)
                if len(pred_full) >= MAX_K:
                    break

        n_valid += 1

        row_out = {
            "row": idx,
            "query": query,
            "idd_laws": ", ".join(sorted(rel_set)),
            "pred_full": ", ".join(pred_full[:MAX_K]),
        }

        for k in K_LIST:
            pred_k = pred_full[:k]
            rr = reciprocal_rank(pred_k, rel_set)
            rec = recall_at_k(pred_k, rel_set)
            ap = average_precision(pred_k, rel_set)
            prec = precision_at_k(pred_k, rel_set)
            f1 = f1_at_k(pred_k, rel_set)

            rr_sum[k] += rr
            rec_sum[k] += rec
            ap_sum[k] += ap
            prec_sum[k] += prec
            f1_sum[k] += f1

            row_out[f"RR@{k}"] = rr
            row_out[f"Recall@{k}"] = rec
            row_out[f"AP@{k}"] = ap
            row_out[f"Precision@{k}"] = prec
            row_out[f"F1@{k}"] = f1

        detail_rows.append(row_out)

    for k in K_LIST:
        mrr = rr_sum[k] / n_valid if n_valid else 0.0
        map_score = ap_sum[k] / n_valid if n_valid else 0.0
        mean_recall = rec_sum[k] / n_valid if n_valid else 0.0
        mean_prec = prec_sum[k] / n_valid if n_valid else 0.0
        mean_f1 = f1_sum[k] / n_valid if n_valid else 0.0

        summary_rows.append({
            "TOPK": k,
            "TotalRows": n_total,
            "Queries": n_valid,
            "MetaFail": n_meta_fail,
            "RetrieveFail": n_retrieve_fail,
            "RerankFail": n_rerank_fail,
            "MRR": round(mrr, 6),
            "MAP": round(map_score, 6),
            "Recall": round(mean_recall, 6),
            "Precision": round(mean_prec, 6),
            "F1": round(mean_f1, 6),
            "CandidateK": CANDIDATE_K,
            "MaxK": MAX_K,
        })

        print(
            f"TOPK={k:>3} | MRR={mrr:.4f} | MAP={map_score:.4f} | "
            f"Recall={mean_recall:.4f} | Precision={mean_prec:.4f} | F1={mean_f1:.4f} | "
            f"Queries={n_valid}/{n_total} | MetaFail={n_meta_fail} | RetFail={n_retrieve_fail} | RerFail={n_rerank_fail}"
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_df.to_excel(writer, sheet_name="summary", index=False)
    detail_df.to_excel(writer, sheet_name="detail_all_k", index=False)
    writer.close()

    print("\n Process is done!")
    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
