from qwen_client import QwenClient


class RetrievalJudge:
    """
    用原来的 LLM 做“是否需要检索”的判断。
    """

    @staticmethod
    def decide_retrieval(user_query: str, history_text: str = "") -> str:
        prompt = f"""
你是一个文档问答系统中的“检索决策器”。

历史对话：
{history_text if history_text.strip() else "无"}

当前问题：
{user_query}

任务：
判断在回答这个问题之前，是否必须先检索上传的文档内容。

判断规则：
1. 如果问题在询问上传文档、文件、论文、报告、合同、PDF、DOCX、TXT中的内容、结论、定义、条款、实验结果、数据、摘要、证据、原文、核心观点，输出 [Retrieve=Yes]
2. 如果问题依赖文档上下文才能准确回答，输出 [Retrieve=Yes]
3. 如果问题只是寒暄、问候、与文档无关的泛化聊天，输出 [Retrieve=No]
4. 只输出一个结果，不要解释

请严格只输出以下两者之一：
[Retrieve=Yes]
[Retrieve=No]
""".strip()

        answer = QwenClient.call(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )

        return "Yes" if "[Retrieve=Yes]" in answer else "No"