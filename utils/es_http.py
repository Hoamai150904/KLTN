import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


ES_BASE_URL = (os.getenv("ES_BASE_URL") or os.getenv("ES_URL") or "").strip().rstrip("/")
ES_URL = ES_BASE_URL
ES_INDEX = (os.getenv("ES_INDEX") or "").strip()
ES_USER = (os.getenv("ES_USER") or "").strip()
ES_PASSWORD = os.getenv("ES_PASSWORD") or ""
ES_API_KEY = (os.getenv("ES_API_KEY") or "").strip()
VERIFY_TLS = _to_bool(os.getenv("ES_VERIFY_TLS"), default=False)


def request_kwargs(timeout: Optional[int] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    req_headers = dict(headers or {})
    kwargs: Dict[str, object] = {"verify": VERIFY_TLS}

    if timeout is not None:
        kwargs["timeout"] = timeout

    if ES_API_KEY:
        req_headers["Authorization"] = f"ApiKey {ES_API_KEY}"
    elif ES_USER:
        kwargs["auth"] = (ES_USER, ES_PASSWORD)

    if req_headers:
        kwargs["headers"] = req_headers

    return kwargs
