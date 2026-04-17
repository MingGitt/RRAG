import time
import asyncio
import requests
from config import QWEN_URL, QWEN_MODEL, HEADERS


class QwenClient:

    @staticmethod
    def extract_answer(resp_json):
        try:
            return resp_json["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    @staticmethod
    def call(messages, temperature=0.0, max_tokens=512, max_retry=5):
        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        last_error_text = ""

        for attempt in range(1, max_retry + 1):
            try:
                r = requests.post(
                    QWEN_URL,
                    headers=HEADERS,
                    json=payload,
                    timeout=60
                )

                if not r.ok:
                    last_error_text = r.text

                if r.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"{r.status_code} Server Error")

                r.raise_for_status()
                return QwenClient.extract_answer(r.json())

            except Exception as e:
                if attempt == max_retry:
                    print("[ERROR] Qwen API failed:", e)
                    if last_error_text:
                        print("[ERROR] response body:", last_error_text[:1000])
                    return ""

                wait = 2 ** attempt
                print(f"[WARN] Qwen retry {attempt}, sleep {wait}s")
                time.sleep(wait)

    @staticmethod
    async def async_call(messages, temperature=0.0, max_tokens=512, max_retry=5):
        """
        异步接口：
        不改原有同步 call 的实现，直接用 asyncio.to_thread 包装。
        这样不需要额外安装 httpx / aiohttp，兼容性最好。
        """
        return await asyncio.to_thread(
            QwenClient.call,
            messages,
            temperature,
            max_tokens,
            max_retry
        )