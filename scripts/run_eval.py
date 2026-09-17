import _bootstrap  # noqa: F401
from financial_rag_agent.eval.run import run_eval, summarize

if __name__ == "__main__":
    results = run_eval()
    report = summarize(results)
    print(report)

    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nWritten to eval_report.md")
