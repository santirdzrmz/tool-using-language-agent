"""
Test cases for the World Cup Stats Agent.

Run:  python test_cases.py
      python test_cases.py react_few_shot   # specify a mode

Saves each trace to the traces/ directory for report analysis.
"""

import json
import os
import sys
from agent import WorldCupAgent

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

SUCCESS_CASES = [
    {
        "id": "S1",
        "label": "All-time top scorer",
        "query": "Who is the all-time top scorer in World Cup history and how many goals did they score?",
        "expect_keywords": ["messi", "19", "klose", "16"],  # Messi (19) surpassed Klose (16) in 2022
    },
    {
        "id": "S2",
        "label": "Country with most titles",
        "query": "Which country has won the most FIFA World Cups and how many times?",
        "expect_keywords": ["brazil", "5"],
    },
    {
        "id": "S3",
        "label": "2022 Golden Boot",
        "query": "Who won the Golden Boot at the 2022 FIFA World Cup in Qatar?",
        "expect_keywords": ["mbappe", "mbappé"],
    },
    {
        "id": "S4",
        "label": "Calculated average goals",
        "query": "France scored 16 goals in 7 matches at the 2018 World Cup. What was their goals-per-game average?",
        "expect_keywords": ["2.28", "2.29", "2.2857", "2.285"],  # 16/7 ≈ 2.2857
    },
    {
        "id": "S5",
        "label": "Historic first World Cup",
        "query": "Which country hosted the first FIFA World Cup in 1930 and who won it?",
        "expect_keywords": ["uruguay"],
    },
]

FAILURE_CASES = [
    {
        "id": "F1",
        "label": "Future tournament (doesn't exist yet)",
        "query": "How many goals were scored in total at the 2026 FIFA World Cup?",
        "expect_failure_signal": ["2026", "not yet", "future", "haven't", "no data", "will be", "upcoming", "cannot", "failed"],
    },
    {
        "id": "F2",
        "label": "Nonsense player name",
        "query": "How many World Cup goals did Zblarg Frumpton score?",
        "expect_failure_signal": ["no", "not found", "doesn't exist", "cannot find", "no record", "unknown"],
    },
    {
        "id": "F3",
        "label": "Division by zero in stat calculation",
        "query": "A team played 0 games but scored 5 goals. Calculate their goals-per-game using 5/0.",
        "expect_failure_signal": ["zero", "undefined", "error", "cannot", "division"],
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(mode: str = "react"):
    agent = WorldCupAgent(mode=mode, max_iterations=10)
    os.makedirs("traces", exist_ok=True)

    results_summary = []

    print(f"\n{'#'*60}")
    print(f"  Running test suite — mode: {mode}")
    print(f"{'#'*60}")

    # ---- Success cases -------------------------------------------------------
    print("\n--- SUCCESS CASES ---")
    for tc in SUCCESS_CASES:
        print(f"\n[{tc['id']}] {tc['label']}")
        try:
            result = agent.run(tc["query"], verbose=True)
        except Exception as e:
            print(f"[ERROR] {e}")
            result = {"answer": str(e), "trace": [], "iterations": 0, "success": False, "mode": mode}

        # Check answer text AND tool outputs in the trace (for calculation tests)
        trace_outputs = " ".join(
            step.get("output", "") for step in result.get("trace", []) if step.get("step") == "tool_use"
        )
        searchable = (result["answer"] + " " + trace_outputs).lower()
        keywords = tc.get("expect_keywords", [])
        passed = any(kw.lower() in searchable for kw in keywords) if keywords else True

        status = "PASS" if (result["success"] and passed) else "FAIL"
        print(f"\n>>> {status} — keywords checked: {keywords}")

        result["test_id"] = tc["id"]
        result["test_label"] = tc["label"]
        result["test_status"] = status
        agent.save_trace(result, f"traces/{tc['id']}_{mode}.json")
        results_summary.append({"id": tc["id"], "label": tc["label"], "status": status})

    # ---- Failure cases -------------------------------------------------------
    print("\n--- FAILURE / EDGE CASES ---")
    for tc in FAILURE_CASES:
        print(f"\n[{tc['id']}] {tc['label']}")
        try:
            result = agent.run(tc["query"], verbose=True)
        except Exception as e:
            print(f"[ERROR] {e}")
            result = {"answer": str(e), "trace": [], "iterations": 0, "success": False, "mode": mode}

        answer_lower = result["answer"].lower()
        signals = tc.get("expect_failure_signal", [])
        # A "pass" here means the agent correctly signalled it can't answer
        passed = any(s.lower() in answer_lower for s in signals) if signals else True

        status = "PASS" if passed else "FAIL"
        print(f"\n>>> {status} — expected signal words: {signals}")

        result["test_id"] = tc["id"]
        result["test_label"] = tc["label"]
        result["test_status"] = status
        agent.save_trace(result, f"traces/{tc['id']}_{mode}.json")
        results_summary.append({"id": tc["id"], "label": tc["label"], "status": status})

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY — mode: {mode}")
    print(f"{'='*60}")
    passes = sum(1 for r in results_summary if r["status"] == "PASS")
    total = len(results_summary)
    for r in results_summary:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  {icon} [{r['id']}] {r['label']}")
    print(f"\n  {passes}/{total} passed")
    print(f"{'='*60}\n")

    return results_summary


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "react"
    run_suite(mode)
