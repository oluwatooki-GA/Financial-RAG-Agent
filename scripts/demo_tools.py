import _bootstrap  # noqa: F401

from financial_rag_agent.tools import calculator
from financial_rag_agent.tools.ngx_corporate_actions import get_corporate_actions
from financial_rag_agent.tools.xbrl import get_company_concept

if __name__ == "__main__":
    print("=== SEC XBRL: real structured figures straight from SEC's API (not scraped HTML) ===")
    facts = get_company_concept("0001045810", "Revenues", taxonomy="us-gaap")
    annual = [f for f in facts if f.fiscal_period == "FY"]
    for f in annual[-3:]:
        print(f"  FY{f.fiscal_year} ({f.end_date}): ${f.value:,.0f} {f.unit}")

    if len(annual) >= 2:
        latest, previous = annual[-1], annual[-2]
        # Note: SEC's "fy" field tags the FILING's fiscal year, not the period
        # the figure covers -- prior-year comparatives share that same tag,
        # which is why end_date (a real distinct period) is used here instead.
        print("\n=== Calculator: growth_rate() on those two real XBRL figures ===")
        growth = calculator.growth_rate(current=latest.value, previous=previous.value)
        print(f"  {previous.end_date} -> {latest.end_date}: {growth:.1%} revenue growth")
        print("  (deterministic math, no LLM involved -- this is the OPERATIONS dict-dispatch tool)")

    print("\n=== NGX corporate actions: real dividend data from ngxgroup.com's public REST API ===")
    actions = get_corporate_actions(2024, company_symbol="GTCO")
    for a in actions:
        print(f"  {a.company}: dividend {a.dividend_share}, AGM {a.agm_date}, paid {a.payment_date}")
