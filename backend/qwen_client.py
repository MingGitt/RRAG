import time
import requests

from config import QWEN_URL, QWEN_MODEL, QW_HEADERS


class QwenClient:
    @staticmethod
    def extract_answer(resp_json):
        # 修改提取答案的逻辑以适应 DashScope 的返回格式
        try:
            # DashScope 返回格式: {"output": {"text": "..."}}
            return resp_json["output"]["text"].strip()
        except Exception:
            # 兼容可能的其他格式
            try:
                return resp_json["choices"][0]["message"]["content"].strip()
            except Exception:
                return ""

    @staticmethod
    def call(messages, temperature=0.0, max_tokens=512, max_retry=5):
        # 修改 payload 格式以符合 DashScope API 要求
        payload = {
            "model": QWEN_MODEL,
            "input": {
                "messages": messages  # messages 必须放在 input 内
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }

        last_error = None

        for attempt in range(1, max_retry + 1):
            try:
                r = requests.post(
                    QWEN_URL,
                    headers=QW_HEADERS,
                    json=payload,
                    timeout=60
                )

                print(f"[QWEN] attempt={attempt}, status_code={r.status_code}")

                if not r.ok:
                    try:
                        error_text = r.text[:1000]
                    except Exception:
                        error_text = "<unable to read response text>"
                    print(f"[QWEN] response_body={error_text}")

                # 不重试的错误
                if r.status_code in [400, 401, 403, 404]:
                    if r.status_code == 404:
                        print("[QWEN][ERROR] model not found or no access. 请检查 QWEN_MODEL 和地域/账号权限。")
                    else:
                        print(f"[QWEN][ERROR] non-retriable error: {r.status_code}")
                    return ""

                # 限流，重试
                if r.status_code == 429:
                    raise requests.exceptions.HTTPError(
                        f"429 Too Many Requests: {r.text[:1000]}"
                    )

                # 服务端错误，重试
                if r.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"{r.status_code} Server Error: {r.text[:1000]}"
                    )

                r.raise_for_status()

                try:
                    resp_json = r.json()
                except Exception as e:
                    print(f"[QWEN][ERROR] json parse failed: {repr(e)}")
                    print(f"[QWEN] raw_text={r.text[:1000]}")
                    return ""

                answer = QwenClient.extract_answer(resp_json)
                if not answer:
                    print("[QWEN][ERROR] empty answer extracted")
                    print(f"[QWEN] resp_json={str(resp_json)[:1000]}")
                    return ""

                return answer

            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"[QWEN][TIMEOUT] attempt={attempt}, error={repr(e)}")

            except requests.exceptions.ConnectionError as e:
                last_error = e
                print(f"[QWEN][CONNECTION_ERROR] attempt={attempt}, error={repr(e)}")

            except requests.exceptions.HTTPError as e:
                last_error = e
                print(f"[QWEN][HTTP_ERROR] attempt={attempt}, error={repr(e)}")

            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"[QWEN][REQUEST_ERROR] attempt={attempt}, error={repr(e)}")

            except Exception as e:
                last_error = e
                print(f"[QWEN][UNKNOWN_ERROR] attempt={attempt}, error={repr(e)}")

            if attempt == max_retry:
                print("[QWEN][FINAL_FAIL] all retries exhausted")
                print(f"[QWEN][FINAL_FAIL] last_error={repr(last_error)}")
                return ""

            wait = min(2 ** attempt, 16)
            print(f"[QWEN][RETRY] sleep {wait}s before next attempt")
            time.sleep(wait)

        return ""