# rag_types.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: Optional[str] = None
    source: Optional[str] = None
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateAnswer:
    text: str
    generation_score: float = 0.0
    evidence_score: float = 0.0
    final_score: float = 0.0
    used_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRiskReport:
    risk_score: float
    risk_level: str
    reasons: List[str]
    should_retry: bool
    suggested_top_k: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRiskReport:
    risk_score: float
    risk_level: str
    reasons: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    rewritten_query: str
    answer: str
    candidates: List[CandidateAnswer]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_risk: RetrievalRiskReport
    generation_risk: GenerationRiskReport
    retry_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)