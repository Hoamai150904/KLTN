import json
from datetime import datetime

import pandas as pd
import requests
import urllib3

# ✅ Tắt cảnh báo verify=False (tùy chọn)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========= CẤU HÌNH =========
ES = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "15092004"
INDEX = "cu_tru_law_v3"

XLSX_PATH = r"M:\Chatbot\data\luattonghop1301.xlsx"
SHEET_NAME = 0

ID_FIELD_PRIORITY = ["idd", "id", "stt"]
DATE_COLUMNS = ["ngay_bh", "ngay_hl", "ngay_hhl"]


# ========= HÀM TIỆN ÍCH =========
def parse_vn_date(x):
    """
    Trả về:
      - 'YYYY-MM-DD' nếu parse được
      - None nếu trống/NaN/NaT/không hợp lệ
    => đảm bảo field date trong ES không bị lỗi vì 'NaT'
    """
    # ✅ Bắt NaN/NaT của pandas (quan trọng nhất)
    if x is None or pd.isna(x):
        return None

    # Nếu đã là datetime/timestamp Excel
    dt = pd.to_datetime(x, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def pick_doc_id(doc: dict):
    for f in ID_FIELD_PRIORITY:
        v = doc.get(f, None)
        if v not in (None, "") and not (isinstance(v, float) and pd.isna(v)):
            return str(v).strip()
    return None


def bulk_index(docs, batch_size=200):
    url = f"{ES}/_bulk?refresh=true"
    headers = {"Content-Type": "application/x-ndjson"}

    total = 0
    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        lines = []

        for doc in chunk:
            # ✅ Chặn tuyệt đối các giá trị "NaT"/"nan" dạng chuỗi lọt vào
            for c in DATE_COLUMNS:
                if c in doc:
                    v = doc[c]
                    if v is None:
                        continue
                    if isinstance(v, str) and v.strip().lower() in ("nat", "nan", ""):
                        doc[c] = None

            doc_id = pick_doc_id(doc)
            action = {"index": {"_index": INDEX}}
            if doc_id:
                action["index"]["_id"] = doc_id

            lines.append(json.dumps(action, ensure_ascii=False))
            lines.append(json.dumps(doc, ensure_ascii=False))

        payload = ("\n".join(lines) + "\n").encode("utf-8")

        r = requests.post(
            url,
            data=payload,
            headers=headers,
            auth=(ES_USER, ES_PASSWORD),
            verify=False,  # self-signed cert
            timeout=120,
        )

        if r.status_code >= 300:
            raise RuntimeError(f"Bulk failed {r.status_code}: {r.text[:2000]}")

        resp = r.json()
        if resp.get("errors"):
            errors = []
            for item in resp.get("items", []):
                op = item.get("index", {})
                if op.get("error"):
                    errors.append(op["error"])
                if len(errors) >= 5:
                    break
            raise RuntimeError(f"Bulk has errors. Sample errors: {errors}")

        total += len(chunk)
        print(f"✅ Indexed {total}/{len(docs)} docs")

    print("🎉 DONE. Total docs indexed:", total)


# ========= MAIN =========
def main():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    df.columns = [c.strip() for c in df.columns]

    # ✅ Chuẩn hóa cột ngày: mọi giá trị không hợp lệ -> None
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_vn_date)

            # ✅ đảm bảo pandas không tự biến None thành NaT ở dtype datetime64
            df[col] = df[col].astype("object")
            df.loc[df[col].isna(), col] = None

    # ✅ Các cột khác: NaN -> None
    df = df.astype("object").where(pd.notnull(df), None)

    docs = df.to_dict(orient="records")

    bulk_index(docs, batch_size=200)


if __name__ == "__main__":
    main()
