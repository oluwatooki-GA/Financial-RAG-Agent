from dataclasses import dataclass


@dataclass
class EvalQuery:
    query: str
    category: str


EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery("What was NVIDIA's total revenue for fiscal year 2026?", "financials"),
    EvalQuery("What are the key risk factors related to supply chain and manufacturing?", "risk"),
    EvalQuery("Describe NVIDIA's Data Center segment performance.", "segments"),
    EvalQuery("What is NVIDIA's Compute & Networking segment operating income?", "segments"),
    EvalQuery("What risks does NVIDIA face related to competition in the semiconductor industry?", "risk"),
    EvalQuery("How much did NVIDIA invest in research and development?", "financials"),
    EvalQuery("What is NVIDIA's Blackwell architecture and how did it affect revenue growth?", "product"),
    EvalQuery("What risks does NVIDIA face related to international operations and export regulations?", "risk"),
    EvalQuery("Describe NVIDIA's revenue by geographic region.", "financials"),
    EvalQuery("What internal controls over financial reporting does NVIDIA maintain?", "governance"),
]
