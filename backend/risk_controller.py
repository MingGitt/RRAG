# risk_controller.py
from typing import Dict, List, Optional


class RiskController:
    """
    风险控制器：
    1. assess_retrieval_risk: 检索/重排后判断检索风险
    2. assess_generation_risk: 生成排序后判断答案风险
    """

    @staticmethod
    def assess_retrieval_risk(ranked_chunks: List[Dict]) -> Dict:
        if not ranked_chunks:
            return {
                "risk_score": 1.0,
                "risk_level": "high",
                "reasons": ["没有检索到任何证据块"],
            }

        scores = [float(item.get("score", 0.0)) for item in ranked_chunks]
        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else 0.0

        risk_score = 0.0
        reasons = []

        if top1 < 0.2:
            risk_score += 0.45
            reasons.append("最高重排分数偏低，证据相关性不足")
        elif top1 < 0.4:
            risk_score += 0.25
            reasons.append("最高重排分数一般，主证据不够强")

        if len(scores) > 1 and abs(top1 - top2) < 0.03:
            risk_score += 0.15
            reasons.append("前两条证据分数接近，主证据不够突出")

        strong_cnt = sum(1 for s in scores[:5] if s >= 0.2)
        if strong_cnt < 2:
            risk_score += 0.25
            reasons.append("高质量证据数量偏少")

        doc_ids = [item.get("doc_id") for item in ranked_chunks[:5] if item.get("doc_id")]
        unique_docs = len(set(doc_ids))
        if len(doc_ids) >= 3 and unique_docs == 1:
            risk_score += 0.10
            reasons.append("证据过于集中在单一文档")

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.6:
            level = "high"
        elif risk_score >= 0.3:
            level = "medium"
        else:
            level = "low"

        if not reasons:
            reasons = ["检索证据质量较稳定"]

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "reasons": reasons,
        }

    @staticmethod
    def assess_generation_risk(candidates: List[Dict], best_candidate: Optional[Dict]) -> Dict:
        if not candidates or not best_candidate:
            return {
                "risk_score": 1.0,
                "risk_level": "high",
                "confidence": 0.0,
                "reasons": ["没有可用的生成答案"],
            }

        risk_score = 0.0
        reasons = []

        evidence_score = float(best_candidate.get("evidence_score", 0.0))
        final_score = float(best_candidate.get("final_score", 0.0))

        if evidence_score < 0.25:
            risk_score += 0.45
            reasons.append("最终答案与证据一致性较弱")
        elif evidence_score < 0.45:
            risk_score += 0.20
            reasons.append("最终答案与证据一致性一般")

        if len(candidates) >= 2:
            gap = float(candidates[0].get("final_score", 0.0)) - float(candidates[1].get("final_score", 0.0))
            if gap < 0.05:
                risk_score += 0.20
                reasons.append("多个候选答案分数接近，结果不够稳定")

        answer = (best_candidate.get("answer") or "").strip()
        if len(answer) < 10:
            risk_score += 0.10
            reasons.append("答案过短，信息可能不足")

        if best_candidate.get("issup") in {"Unknown", "No Support"}:
            risk_score += 0.15
            reasons.append("Self-RAG 反思信号显示支持度不足")

        if len(best_candidate.get("used_chunk_ids", [])) < 1:
            risk_score += 0.10
            reasons.append("答案缺少明显的证据支撑块")

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.6:
            level = "high"
        elif risk_score >= 0.3:
            level = "medium"
        else:
            level = "low"

        confidence = round(max(0.0, 1.0 - risk_score), 4)

        if not reasons:
            reasons = ["答案与证据整体一致，结果较稳定"]

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "confidence": confidence,
            "reasons": reasons,
            "best_final_score": round(final_score, 4),
            "best_evidence_score": round(evidence_score, 4),
        }