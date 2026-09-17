import logging
import time

import _bootstrap  # noqa: F401
from financial_rag_agent.eval.run import run_eval, summarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval_run.log", mode="w", encoding="utf-8"),
    ],
)

if __name__ == "__main__":
    start = time.perf_counter()
    results = run_eval()
    logging.getLogger(__name__).info("Total eval run time: %.1fs", time.perf_counter() - start)

    report = summarize(results)
    print(report)

    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nWritten to eval_report.md (progress/timing log: eval_run.log)")
