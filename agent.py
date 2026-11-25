"""
World Cup Stats Agent
---------------------
A language agent that answers FIFA World Cup questions using four tools:
  - web_search        : DuckDuckGo search for stats and news
  - wikipedia_search  : Wikipedia article lookups
  - calculate_stat    : Arithmetic for derived statistics
  - fetch_url         : Read the full content of a specific web page

Three reasoning modes can be compared:
  "zero_shot"       - minimal system prompt, model decides how to use tools
  "react"           - explicit Thought / Action / Observation structure
  "react_few_shot"  - ReAct + two worked examples to anchor the format

Powered by Groq (free tier). Requires GROQ_API_KEY in environment or .env file.
Get a free key at console.groq.com — no credit card required.
"""

import json
import os
import time
from groq import Groq
from groq import RateLimitError, InternalServerError, BadRequestError
from datetime import datetime
from dotenv import load_dotenv
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

load_dotenv()

# ---------------------------------------------------------------------------
# System prompts – one per reasoning mode
# ---------------------------------------------------------------------------

_BASE_CONTEXT = (
    "You are a specialist FIFA World Cup statistics analyst. "
    "You have deep knowledge of all World Cup tournaments from 1930 to 2022, "
    "including scores, scorers, team records, and historical facts. "
    "Always cite where your numbers come from. "
    "When you compute a statistic using a tool, you MUST state the exact numeric result in your final answer. "
    "If a question cannot be answered with available data, say so clearly."
)

SYSTEM_PROMPTS = {
    # --- Mode 1: zero-shot ---------------------------------------------------
    "zero_shot": (
        f"{_BASE_CONTEXT}\n\n"
        "Use the available tools to look up information and calculate statistics. "
        "Answer clearly and concisely once you have enough data."
    ),

    # --- Mode 2: ReAct -------------------------------------------------------
    "react": (
        f"{_BASE_CONTEXT}\n\n"
        "Use a step-by-step reasoning approach to answer every question:\n"
        "  1. Think about what information you need.\n"
        "  2. Use the available tools to look it up.\n"
        "  3. Read the result and decide if you need more information.\n"
        "  4. Repeat until you have enough, then give a clear final answer.\n\n"
        "Be systematic: decompose multi-part questions, cross-check figures from "
        "multiple sources when they seem uncertain, and always show your reasoning."
    ),

    # --- Mode 3: ReAct + few-shot examples -----------------------------------
    "react_few_shot": (
        f"{_BASE_CONTEXT}\n\n"
        "Use a step-by-step reasoning approach. Think, look up, observe, repeat, then answer.\n\n"
        "Example 1:\n"
        "User: Who is the all-time top scorer in World Cup history?\n"
        "Reasoning: I need the all-time World Cup scoring record. Wikipedia likely has this.\n"
        "[calls wikipedia_search: 'FIFA World Cup all-time top scorers']\n"
        "Result: Miroslav Klose holds the record with 16 goals across 4 tournaments.\n"
        "Answer: Miroslav Klose (Germany) is the all-time top scorer with 16 World Cup goals, "
        "scored across 2002, 2006, 2010, and 2014.\n\n"
        "Example 2:\n"
        "User: What was Brazil's average goals per game at the 2002 World Cup?\n"
        "Reasoning: I need Brazil's total goals and games played in 2002.\n"
        "[calls wikipedia_search: 'Brazil 2002 FIFA World Cup']\n"
        "Result: Brazil scored 18 goals in 7 matches to win the tournament.\n"
        "Reasoning: Now I can compute the average.\n"
        "[calls calculate_stat: '18 / 7']\n"
        "Result: 2.5714\n"
        "Answer: Brazil averaged 2.57 goals per game at the 2002 World Cup (18 goals in 7 matches).\n\n"
        "Now answer the user's question using the same approach."
    ),
}

# ---------------------------------------------------------------------------
# Convert Anthropic-style tool definitions → OpenAI/Groq format
# ---------------------------------------------------------------------------

def _build_groq_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in TOOL_DEFINITIONS
    ]


GROQ_TOOLS = _build_groq_tools()

# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class WorldCupAgent:
    """
    Agentic loop that calls Groq with tool use until the model
    produces a final text response (no more tool calls).
    """

    MODEL = "llama-3.1-8b-instant"

    def __init__(self, mode: str = "react", max_iterations: int = 10):
        if mode not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(SYSTEM_PROMPTS)}")
        self.mode = mode
        self.max_iterations = max_iterations
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    def _call_with_retry(self, messages, max_retries: int = 5):
        """Call the Groq API, retrying on rate-limit and server errors."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    tools=GROQ_TOOLS,
                    tool_choice="auto",
                    max_tokens=4096,
                )
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait = 30
                    print(f"\n[Rate limit — waiting {wait}s before retry {attempt + 1}/{max_retries - 1}]")
                    time.sleep(wait)
                else:
                    raise
            except InternalServerError as e:
                if attempt < max_retries - 1:
                    wait = 15
                    print(f"\n[Server error — waiting {wait}s before retry {attempt + 1}/{max_retries - 1}]")
                    time.sleep(wait)
                else:
                    raise
            except BadRequestError as e:
                # Model generated a malformed tool call — fall back to a
                # no-tools call so it answers from parametric knowledge
                print(f"\n[Bad tool call — falling back to direct answer]")
                try:
                    return self.client.chat.completions.create(
                        model=self.MODEL,
                        messages=messages,
                        max_tokens=4096,
                    )
                except Exception:
                    raise e  # re-raise the original error if fallback also fails

    # ------------------------------------------------------------------
    def run(self, query: str, verbose: bool = True) -> dict:
        """
        Run the agent on a query and return a result dict with:
          - answer        : final text answer
          - trace         : list of step dicts for analysis
          - iterations    : number of tool-use rounds
          - success       : True if a final answer was reached
          - mode          : reasoning mode used
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[self.mode]},
            {"role": "user",   "content": query},
        ]
        trace = []
        iterations = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query  : {query}")
            print(f"Mode   : {self.mode}")
            print(f"{'='*60}")

        for _ in range(self.max_iterations):
            try:
                response = self._call_with_retry(messages)
            except BadRequestError as e:
                print(f"\n[Unrecoverable bad request — ending run]")
                return {
                    "answer": str(e),
                    "trace": trace,
                    "iterations": iterations,
                    "success": False,
                    "mode": self.mode,
                }
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            text = message.content or ""

            if verbose and text:
                print(f"\n[Model]: {text}")

            # ---- No tool calls → final answer --------------------------
            if not tool_calls:
                trace.append({"step": "final_answer", "content": text})
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Final Answer:\n{text}")
                    print(f"{'='*60}\n")
                return {
                    "answer": text,
                    "trace": trace,
                    "iterations": iterations,
                    "success": bool(text),
                    "mode": self.mode,
                }

            # ---- Execute tool calls ------------------------------------
            # Append assistant message with tool_calls
            messages.append({"role": "assistant", "content": text, "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]})

            for tc in tool_calls:
                tool_name  = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                iterations += 1

                if verbose:
                    print(f"\n[Tool call]: {tool_name}({json.dumps(tool_input)})")

                func = TOOL_FUNCTIONS.get(tool_name)
                if func is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = func(**tool_input)
                    except Exception as e:
                        result = f"Tool error: {e}"

                if verbose:
                    preview = result[:300] + ("..." if len(result) > 300 else "")
                    print(f"[Observation]: {preview}")

                trace.append({
                    "step":   "tool_use",
                    "tool":   tool_name,
                    "input":  tool_input,
                    "output": result,
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        # Exceeded max iterations
        return {
            "answer": "",
            "trace": trace,
            "iterations": iterations,
            "success": False,
            "mode": self.mode,
            "error": f"Reached max iterations ({self.max_iterations}) without a final answer.",
        }

    # ------------------------------------------------------------------
    def save_trace(self, result: dict, filepath: str = None):
        """Save an agent result dict to a JSON file for later analysis."""
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"traces/trace_{self.mode}_{ts}.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Trace saved to {filepath}")


# ---------------------------------------------------------------------------
# Quick interactive mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "react"
    agent = WorldCupAgent(mode=mode)

    print("World Cup Stats Agent  (type 'quit' to exit)")
    print(f"Reasoning mode: {mode}\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        agent.run(query, verbose=True)
