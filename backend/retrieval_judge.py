# retrieval_judge.py
from config import QWEN_DECISION_MODEL
from qwen_client import QwenClient


class RetrievalJudge:
    """
    Step 1: 检索决策
    输出：
      [Retrieve=Yes]
      [Retrieve=No]
    """

    @staticmethod
    def decide_retrieval(user_query: str, history_text: str = "") -> str:
        prompt = f"""
你是一个文档智能检索系统中的“检索决策器”。

历史对话：
{history_text if history_text.strip() else "无"}

当前用户查询：
{user_query}

任务：
判断这个问题在回答之前是否必须先检索上传文档中的内容。

判断规则：
1. 如果问题询问上传文档、论文、报告、合同、PDF、DOCX、TXT中的事实、条款、实验结果、结论、证据、定义、摘要、原文内容，输出 [Retrieve=Yes]
2. 如果问题依赖文档上下文才能准确回答，输出 [Retrieve=Yes]
3. 如果问题是泛化闲聊、通用常识、与文档无关，输出 [Retrieve=No]
4. 只能输出一个结果，不要解释

请严格只输出：
[Retrieve=Yes]
或
[Retrieve=No]
""".strip()

        answer = QwenClient.call(
            [{"role": "user", "content": prompt}],
            model=QWEN_DECISION_MODEL,
            temperature=0.0,
            max_tokens=32,
        )

        return "Yes" if "[Retrieve=Yes]" in answer else "No"