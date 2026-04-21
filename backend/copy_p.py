from selfrag_llm import SelfRagLLM

llm = SelfRagLLM()
text = llm.generate_with_evidence(
    query="Aspirin inhibits the production of PGE2.",
    evidence_text="Aspirin inhibits cyclooxygenase and reduces the production of prostaglandin E2 (PGE2)."
)
print(text)