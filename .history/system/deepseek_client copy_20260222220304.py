import time
import requests
from config import DEEPSEEK_URL, DEEPSEEK_MODEL, HEADERS

class DeepSeekClient:

    @staticmethod
    def extract_answer(resp_json):
        choice = resp_json["choices"][0]["message"]
        if choice.get("content"):
            return choice["content"].strip()
        if choice.get("reasoning_content"):
            return choice["reasoning_content"].strip()
        return ""

    @staticmethod
    def call(messages, temperature=0.0, max_tokens=256, max_retry=5):
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, max_retry + 1):
            try:
                r = requests.post(DEEPSEEK_URL, headers=HEADERS, json=payload, timeout=60)

                if r.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"{r.status_code} Server Error")

                r.raise_for_status()
                return DeepSeekClient.extract_answer(r.json())

            except Exception as e:
                if attempt == max_retry:
                    print("[ERROR] DeepSeek failed:", e)
                    return ""

                wait = 2 ** attempt
                print(f"[WARN] retry {attempt}, sleep {wait}s")
                time.sleep(wait)