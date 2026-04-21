import time
import requests

from config import QWEN_URL, QWEN_MODEL, QW_HEADERS, QWEN_REQUEST_TIMEOUT


class QwenClient:
    """
    用原来的 Qwen API 做通用对话 / 分类判断。
    """

    @staticmethod
    def extract_answer(resp_json):
        try:
            return resp_json["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    @staticmethod
    def call(messages, temperature=0.0, max_tokens=256, max_retry=3):
        if not QWEN_URL:
            print("[QWEN][ERROR] QWEN_URL 未配置")
            return ""

        if not QW_HEADERS.get("Authorization"):
            print("[QWEN][ERROR] DASHSCOPE_API_KEY 未配置")
            return ""

        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None

        for attempt in range(1, max_retry + 1):
            try:
                resp = requests.post(
                    QWEN_URL,
                    headers=QW_HEADERS,
                    json=payload,
                    timeout=QWEN_REQUEST_TIMEOUT,
                )

                print(f"[QWEN] attempt={attempt}, status_code={resp.status_code}")

                if not resp.ok:
                    try:
                        print("[QWEN] response_body=", resp.text[:2000])
                    except Exception:
                        pass

                resp.raise_for_status()
                data = resp.json()

                answer = QwenClient.extract_answer(data)
                if answer:
                    return answer

                print("[QWEN][ERROR] empty answer extracted")
                return ""

            except Exception as e:
                last_error = e
                print(f"[QWEN][ERROR] attempt={attempt}, error={repr(e)}")
                if attempt < max_retry:
                    time.sleep(min(2 ** attempt, 8))

        print(f"[QWEN][FINAL_FAIL] last_error={repr(last_error)}")
        return ""