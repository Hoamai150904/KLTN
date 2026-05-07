from datetime import date, datetime
from typing import Dict, Any

DEFAULT_WEIGHTS = {
    "title": 2.2,
    "title_dk": 2.0,
    "title_law": 1.8,
    "content_use": 1.3,
    "content": 1.0,
    "content_raw": 0.8,
    "idd": 0.6,
    "law": 0.6,
    "idsdbs": 0.2,
    "link_online": 0.1,
    "source_file": 0.1,
    "ngay_bh": 0.0,
    "ngay_hl": 0.0,
    "ngay_hhl": 0.0,
}


def _to_iso_date(x):
    if x is None:
        return None

    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()

    s = str(x).strip()
    if not s or s.lower() in ("nan", "nat", "none", "null"):
        return None

    head = s[:10]
    if len(head) == 10 and head[4] == "-" and head[7] == "-":
        try:
            datetime.strptime(head, "%Y-%m-%d")
            return head
        except ValueError:
            pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y.%m.%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue

    return None

def get_reference_date(meta: Dict[str, Any]):
    for key in ("date_hint", "post_date"):
        v = _to_iso_date(meta.get(key))
        if v:
            return v
    return date.today().isoformat()

def build_effective_filter(meta: Dict[str, Any], reference_date: str):
    status = meta.get("effective_status", "con_hieu_luc")

    if status in ("bat_ky", "unknown"):
        return []

    if status == "con_hieu_luc":
        return [
            {
                "bool": {
                    "should": [
                        {"range": {"ngay_hl": {"lte": reference_date}}},
                        {"bool": {"must_not": [{"exists": {"field": "ngay_hl"}}]}}
                    ],
                    "minimum_should_match": 1
                }
            },
            {
                "bool": {
                    "should": [
                        {"range": {"ngay_hhl": {"gte": reference_date}}},
                        {"bool": {"must_not": [{"exists": {"field": "ngay_hhl"}}]}}
                    ],
                    "minimum_should_match": 1
                }
            }
        ]

    if status == "het_hieu_luc":
        return [{"range": {"ngay_hhl": {"lt": reference_date}}}]

    return []