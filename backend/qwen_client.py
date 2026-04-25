# qwen_client.py
import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from config import QWEN_URL, QWEN_MODEL, QW_HEADERS, QWEN_REQUEST_TIMEOUT


class QwenClient:
    """
    通用 Qwen 调用器
    支持：
    - 指定 model
    - 普通文本返回
    - JSON 风格返回
    """

    @staticmethod
    def extract_answer(resp_json):
        try:
            return resp_json["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    @staticmethod
    def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
        """
        从模型输出中尽量提取第一个 JSON 对象
        """
        if not text:
            return None

        text = text.strip()

        # 去 markdown code fence
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        # 直接整体 parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 提取首个 {...}
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        return None

    @staticmethod
    def call(
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retry: int = 3,
    ) -> str:
        if not QWEN_URL:
            print("[QWEN][ERROR] QWEN_URL 未配置")
            return ""

        if not QW_HEADERS.get("Authorization"):
            print("[QWEN][ERROR] DASHSCOPE_API_KEY 未配置")
            return ""

        payload = {
            "model": model or QWEN_MODEL,
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

                print(f"[QWEN] model={payload['model']} attempt={attempt}, status_code={resp.status_code}")

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
                print(f"[QWEN][ERROR] model={payload['model']} attempt={attempt}, error={repr(e)}")
                if attempt < max_retry:
                    time.sleep(min(2 ** attempt, 8))

        print(f"[QWEN][FINAL_FAIL] model={payload['model']} last_error={repr(last_error)}")
        return ""

    @staticmethod
    def call_json(
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = QwenClient.call(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = QwenClient._extract_json_block(text)
        if data is not None:
            return data
        return default or {}