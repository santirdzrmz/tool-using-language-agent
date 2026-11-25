# World Cup Stats Agent

A tool-using language agent that answers questions about FIFA World Cup statistics: historical records, tournament results, top scorers, country performance and derived metrics such as goals per game. When a question can't be answered (a future tournament, a player who doesn't exist, undefined arithmetic) it says so instead of guessing.

Built for the Learning & NLP course at the Universitat Autònoma de Barcelona (June 2026).

## How it works

```
User question
      │
      ▼
 WorldCupAgent
  ├── System prompt  (zero_shot / react / react_few_shot)
  └── Agentic loop   (Groq llama-3.1-8b-instant + tool calling)
        ├── web_search       → DuckDuckGo live search
        ├── wikipedia_search → MediaWiki API
        ├── calculate_stat   → safe arithmetic evaluation
        └── fetch_url        → requests + BeautifulSoup
```

The loop runs until the model answers without calling a tool, or for at most 10 iterations. If the model emits a malformed tool call, the agent falls back to a no-tools request so it never crashes.

## Results

All 5 success cases and all 3 edge cases pass. For example, the agent correctly reports Messi (19 goals) as the all-time top scorer, refuses to invent statistics for the 2026 tournament or a made-up player, and reports 5/0 as undefined. The full analysis is in `report.ipynb`, which compares the three prompting strategies and walks through the execution traces saved in `traces/`.

## Files

| File | What it is |
|---|---|
| `agent.py` | The agent loop and the three system-prompt strategies |
| `tools.py` | The four tools |
| `test_cases.py` | Success and failure test queries |
| `report.ipynb` | Report: specification, tools, reasoning comparison, trace analysis |
| `traces/` | Saved execution traces for every test |

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your Groq API key
python test_cases.py
```
