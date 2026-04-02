import os
import json
from typing import Dict, Any, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL

client = OpenAI(**client_kwargs)

MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# =========================
# DEFAULT BOOSTS (GIỮ TRIẾT LÝ CŨ)
# =========================
DEFAULT_WEIGHTS = {
    "title_dk": 0.95,
    "title_law": 0.95,
    "content_use": 0.7,
    "content": 0.7,
    "content_raw": 0.7,
    "idd": 0.3,
    "law": 0.3,
    "idsdbs": 0.2,
    "source_file": 0.1,
    "link_online": 0.1,
    "ngay_bh": 0.0,
    "ngay_hl": 0.0,
    "ngay_hhl": 0.0,
}

# =========================
# SYSTEM PROMPT (BƯỚC 2 - PHỔ THÔNG FIRST)
# =========================
SYSTEM = """
Bạn là hệ thống trích xuất metadata cho truy vấn pháp luật CƯ TRÚ tại Việt Nam.

⚠️ BẮT BUỘC:
- Chỉ trả về JSON hợp lệ (một object)
- Không markdown
- Không giải thích
- Không thêm text ngoài JSON

Mục tiêu: người dùng thường KHÔNG nói "Điều/Chương/Luật". Hãy ưu tiên:
(1) nhận diện ý định (intent)
(2) suy ra chủ đề cư trú (topic)
(3) trích xuất entities đời thường (actor/relationship/scenario/location)
(4) chuẩn hoá từ ngữ đời thường -> thuật ngữ pháp lý (keyword_normalized)
(5) viết lại truy vấn (query_rewrite) theo ngôn ngữ pháp lý để tăng recall.

====================
A) intent (chỉ 1):
- "thu_tuc"        (cần giấy tờ gì, nộp ở đâu, làm thế nào...)
- "dieu_kien"      (khi nào được, điều kiện gì...)
- "xu_phat"        (phạt bao nhiêu, bị xử lý thế nào...)
- "dinh_nghia"     (khái niệm là gì...)
- "tham_quyen"     (cơ quan nào giải quyết...)
- "quyen_nghia_vu" (có được không, phải không...)
- "cam_ket_xac_nhan"     (xin/cấp/xác nhận thông tin cư trú, giấy xác nhận cư trú...)
- "khai_bao_thong_bao"   (khai báo/ thông báo lưu trú, khai báo tạm vắng, khai báo tạm trú...)
- "unknown"

====================
B) topic_candidates (tối đa 3, ưu tiên đúng nhất trước):
- "thuong_tru"
- "tam_tru"
- "tam_vang"
- "luu_tru"
- "xac_nhan_thong_tin_cu_tru"
- "tach_ho_nhap_ho"
- "xoa_dang_ky_cu_tru"
- "dieu_chinh_thong_tin"
- "noi_cu_tru_nguoi_chua_thanh_nien"
- "unknown"

====================
C) entities (object)
Trả về object entities với các field (được phép rỗng/null):
- actor: ["nguoi_thue_tro", "chu_ho", "nguoi_o_nho", "nguoi_chua_thanh_nien", "cong_dan_vn", "nguoi_nuoc_ngoai", ...]
- relationship: ["vo_chong", "cha_me_con", "giam_ho", ...]
- scenario: ["chuyen_noi_o", "o_tro", "o_khach_san", "ve_que", "di_lam_xa", "nhap_ho", "tach_ho", ...]
- location: ["cung_tinh", "khac_tinh", "unknown"]
Nếu không chắc: dùng [] hoặc "unknown".

====================
D) keyword normalization (rất quan trọng)
Trả về:
- keywords_original: danh sách cụm từ quan trọng giữ nguyên lời người dùng
- keyword_normalized: danh sách thuật ngữ pháp lý tương ứng

Ví dụ mapping phổ biến:
- "nhập hộ khẩu" -> "đăng ký thường trú"
- "cắt hộ khẩu" -> "xóa đăng ký thường trú" hoặc "tách hộ"
- "KT3" -> "tạm trú"
- "khai báo qua đêm" -> "thông báo lưu trú"
- "giấy xác nhận cư trú" -> "xác nhận thông tin về cư trú"
- "con theo cha/mẹ" -> "nơi cư trú của người chưa thành niên"

====================
E) needs_currentness & prefer_version
- needs_currentness: true nếu người dùng nói "hiện nay/hiện hành/mới nhất/đang áp dụng/bây giờ..."
- prefer_version:
  - "merged" nếu needs_currentness=true (ưu tiên content_use nếu có)
  - "original" nếu người dùng nói "gốc/ban đầu/trước sửa đổi/nguyên bản..."
  - "both" nếu không rõ

====================
F) Tương thích pipeline cũ:
Bạn vẫn phải xuất các field cũ để hệ thống BM25 dùng được:
- procedure_type (chỉ 1): "tam_tru" | "thuong_tru" | "tam_vang" | "luu_tru" | "unknown"
  => suy ra từ topic_candidates[0] nếu có
- action (chỉ 1 – theo procedure_type; nếu không rõ -> "unknown"):
  TAM_TRU: dang_ky | cap_the | gia_han | xoa | dieu_chinh
  THUONG_TRU: dang_ky | xoa | cap_the | chuyen_dang_ky | dieu_chinh | gia_han
  TAM_VANG: khai_bao | dang_ky_nvqs
  LUU_TRU: thong_bao
- subject: "vn_gt_14" | "vn_lt_14" | "nuoc_ngoai" (mặc định vn_gt_14)
- context: "noi_dia_ngan" | "noi_dia_dai" | "xuat_canh_ngan" | "xuat_canh_dai" | "unknown"
- effective_status: "con_hieu_luc" | "het_hieu_luc" | "bat_ky" | "unknown"
- doc_type: string hoặc "unknown"
- must_keywords / should_keywords / exclude_keywords: danh sách string
- law_numbers: danh sách string (có thể rỗng)
- date_hint: "YYYY-MM-DD" hoặc null

====================
G) Output JSON fields (đủ, đúng tên):
intent,
topic_candidates,
entities,
keywords_original,
keyword_normalized,
query_rewrite,
needs_currentness,
prefer_version,

procedure_type,
action,
subject,
context,
doc_type,
must_keywords,
should_keywords,
exclude_keywords,
law_numbers,
effective_status,
date_hint
"""

# =========================
# Helpers
# =========================
def attach_default_boosts(meta: Dict[str, Any]) -> Dict[str, Any]:
    meta.pop("boost", None)
    meta.pop("boosts", None)
    meta["boosts"] = dict(DEFAULT_WEIGHTS)
    return meta


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x).strip()] if str(x).strip() else []


def _as_bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("true", "1", "yes", "y"):
            return True
        if t in ("false", "0", "no", "n"):
            return False
    return default


def _safe_get_str(meta: Dict[str, Any], key: str, default: str = "unknown") -> str:
    v = meta.get(key, default)
    if v is None:
        return default
    if isinstance(v, str):
        v = v.strip()
        return v if v else default
    return str(v)


def normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hoá output để:
    - không thiếu field (tránh KeyError downstream)
    - giữ tương thích code cũ (procedure_type/action/keywords...)
    - ưu tiên phổ thông-first (entities, query_rewrite, keyword_normalized...)
    """
    if not isinstance(meta, dict):
        raise RuntimeError("Model output must be a JSON object")

    # ---------- New fields (phổ thông-first)
    meta["intent"] = _safe_get_str(meta, "intent", "unknown")
    allowed_intents = {
        "thu_tuc", "dieu_kien", "xu_phat", "dinh_nghia",
        "tham_quyen", "quyen_nghia_vu",
        "cam_ket_xac_nhan", "khai_bao_thong_bao",
        "unknown"
    }
    if meta["intent"] not in allowed_intents:
        meta["intent"] = "unknown"

    meta["topic_candidates"] = _as_list(meta.get("topic_candidates"))

    # Entities normalize (luôn có object)
    entities = meta.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    entities_norm = {
        "actor": _as_list(entities.get("actor")),
        "relationship": _as_list(entities.get("relationship")),
        "scenario": _as_list(entities.get("scenario")),
        "location": _as_list(entities.get("location")),
    }
    # normalize location value if present
    if entities_norm["location"]:
        loc = entities_norm["location"][0].strip().lower()
        if loc not in ("cung_tinh", "khac_tinh", "unknown"):
            entities_norm["location"] = ["unknown"]
        else:
            entities_norm["location"] = [loc]
    meta["entities"] = entities_norm

    meta["keywords_original"] = _as_list(meta.get("keywords_original"))
    meta["keyword_normalized"] = _as_list(meta.get("keyword_normalized"))
    meta["query_rewrite"] = _as_list(meta.get("query_rewrite"))  # 1-2 câu
    meta["needs_currentness"] = _as_bool(meta.get("needs_currentness"), default=False)

    prefer_version = _safe_get_str(meta, "prefer_version", "both").lower()
    if prefer_version not in ("merged", "original", "both"):
        prefer_version = "both"
    # Auto rule: nếu needs_currentness=true mà prefer_version lạc thì ép về merged
    if meta["needs_currentness"] and prefer_version == "both":
        prefer_version = "merged"
    meta["prefer_version"] = prefer_version

    # ---------- Old fields (tương thích pipeline cũ)
    procedure_type = _safe_get_str(meta, "procedure_type", "unknown").lower()
    if procedure_type not in ("tam_tru", "thuong_tru", "tam_vang", "luu_tru", "unknown"):
        procedure_type = "unknown"

    # Nếu model không set procedure_type, suy ra từ topic_candidates[0]
    if procedure_type == "unknown" and meta["topic_candidates"]:
        top_topic = meta["topic_candidates"][0].lower()
        if top_topic in ("tam_tru", "thuong_tru", "tam_vang", "luu_tru"):
            procedure_type = top_topic
    meta["procedure_type"] = procedure_type

    meta["action"] = _safe_get_str(meta, "action", "unknown").lower()

    meta["subject"] = _safe_get_str(meta, "subject", "vn_gt_14")
    if meta["subject"] not in ("vn_gt_14", "vn_lt_14", "nuoc_ngoai"):
        meta["subject"] = "vn_gt_14"

    meta["context"] = _safe_get_str(meta, "context", "unknown")
    if meta["context"] not in ("noi_dia_ngan", "noi_dia_dai", "xuat_canh_ngan", "xuat_canh_dai", "unknown"):
        meta["context"] = "unknown"

    meta["doc_type"] = _safe_get_str(meta, "doc_type", "unknown")

    meta["must_keywords"] = _as_list(meta.get("must_keywords"))
    meta["should_keywords"] = _as_list(meta.get("should_keywords"))
    meta["exclude_keywords"] = _as_list(meta.get("exclude_keywords"))
    meta["law_numbers"] = _as_list(meta.get("law_numbers"))

    meta["effective_status"] = _safe_get_str(meta, "effective_status", "unknown")
    if meta["effective_status"] not in ("con_hieu_luc", "het_hieu_luc", "bat_ky", "unknown"):
        meta["effective_status"] = "unknown"

    date_hint = meta.get("date_hint", None)
    if isinstance(date_hint, str):
        date_hint = date_hint.strip() or None
    meta["date_hint"] = date_hint

    # ---- Boost keywords: nếu chưa có must/should mà có normalized/rewrite thì tự bơm để retrieval ăn ngay
    if not meta["must_keywords"] and not meta["should_keywords"]:
        meta["should_keywords"] = list(dict.fromkeys(meta["keyword_normalized"] + meta["query_rewrite"]))

    if not meta["should_keywords"] and meta["keywords_original"]:
        meta["should_keywords"] = list(dict.fromkeys(meta["keywords_original"]))

    return meta


# =========================
# MAIN
# =========================
def extract_metadata(question: str) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return attach_default_boosts(normalize_meta({
            "intent": "unknown",
            "topic_candidates": [],
            "entities": {"actor": [], "relationship": [], "scenario": [], "location": ["unknown"]},
            "keywords_original": [],
            "keyword_normalized": [],
            "query_rewrite": [],
            "needs_currentness": False,
            "prefer_version": "both",

            "procedure_type": "unknown",
            "action": "unknown",
            "subject": "vn_gt_14",
            "context": "unknown",
            "doc_type": "unknown",
            "must_keywords": [],
            "should_keywords": [],
            "exclude_keywords": [],
            "law_numbers": [],
            "effective_status": "unknown",
            "date_hint": None,
        }))

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ],
    )

    text = (resp.output_text or "").strip()

    try:
        meta = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            meta = json.loads(text[start:end + 1])
        else:
            raise RuntimeError(f"LLM output is not valid JSON. Raw: {text[:300]}")

    meta = normalize_meta(meta)
    return attach_default_boosts(meta)


if __name__ == "__main__":
    tests = [
        "Con theo cha hay theo mẹ thì đăng ký cư trú thế nào?",
        "Ở trọ có cần khai báo gì không? KT3 là gì?",
        "Hiện nay muốn nhập hộ khẩu về nhà chồng cần giấy tờ gì?",
        "Không đăng ký tạm trú bị phạt bao nhiêu?",
        "Giấy xác nhận cư trú xin ở đâu?"
    ]
    for q in tests:
        print("Q:", q)
        print(json.dumps(extract_metadata(q), ensure_ascii=False, indent=2))
        print("-" * 80)
