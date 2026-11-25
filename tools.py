"""
Tools for the World Cup Stats Agent.
Each function is called by the agent when it needs external data or computation.
"""

import math
import re
import requests
from bs4 import BeautifulSoup


def web_search(query: str) -> str:
    """Search the web for World Cup stats and news."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        # Bias queries toward football/soccer context
        football_query = f"{query} FIFA World Cup football"
        with DDGS() as ddgs:
            results = list(ddgs.text(football_query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')}"
            )
        return "\n\n".join(lines)

    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Error during web search: {e}"


def wikipedia_search(topic: str) -> str:
    """
    Look up a World Cup topic on Wikipedia.
    Good for historical records, tournament summaries, player bios, and team histories.
    """
    try:
        api = "https://en.wikipedia.org/w/api.php"

        # Find the best matching page title
        search_resp = requests.get(
            api,
            params={"action": "query", "list": "search", "srsearch": topic,
                    "format": "json", "srlimit": 1},
            timeout=10,
        )
        search_data = search_resp.json()
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia article found for: {topic}"

        page_title = results[0]["title"]

        # Fetch the article intro
        extract_resp = requests.get(
            api,
            params={"action": "query", "prop": "extracts", "exintro": True,
                    "explaintext": True, "titles": page_title, "format": "json"},
            timeout=10,
        )
        extract_data = extract_resp.json()
        pages = extract_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()

        if not extract:
            return f"Wikipedia article '{page_title}' has no text extract."

        # Keep first 2000 chars to stay within context limits
        if len(extract) > 2000:
            extract = extract[:2000] + "\n[...article truncated]"

        return f"Wikipedia — {page_title}\n\n{extract}"

    except Exception as e:
        return f"Error querying Wikipedia: {e}"


def calculate_stat(expression: str) -> str:
    """
    Evaluate a mathematical expression to compute derived football statistics.
    Examples: goals per game, win percentage, goal difference, tournament averages.
    Supports: +, -, *, /, **, sqrt(), round(), abs(), floor(), ceil().
    """
    # Block anything dangerous before eval
    if re.search(r"(import|exec|eval|open|__)", expression):
        return "Error: expression contains disallowed keywords."

    # Normalize unicode operators and whitespace (LLMs sometimes emit lookalike chars)
    _unicode_map = {
        '÷': '/', '∕': '/', '／': '/', '⁄': '/',  # division variants
        '×': '*', '∗': '*',                                  # multiplication variants
        '−': '-', '–': '-', '—': '-',                  # minus variants
        ' ': ' ', ' ': ' ', ' ': ' ',                  # space variants
    }
    expression = expression.translate(str.maketrans(_unicode_map))
    expr = " ".join(expression.split())
    expr = expr.replace("^", "**")  # must be a plain replace, not regex (^ is a regex anchor)
    word_replacements = {
        "sqrt": "math.sqrt",
        "log10": "math.log10",
        "log": "math.log",
        "sin": "math.sin",
        "cos": "math.cos",
        "floor": "math.floor",
        "ceil": "math.ceil",
        "abs": "abs",
        "round": "round",
        "pi": "math.pi",
    }
    for old, new in word_replacements.items():
        expr = re.sub(rf"\b{old}\b", new, expr)

    try:
        result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs, "round": round})
        # Format nicely: drop unnecessary decimals
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.4f}".rstrip("0").rstrip(".")
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error evaluating expression: {e}"


def fetch_url(url: str) -> str:
    """
    Fetch the text content of a web page.
    Use this to read a specific stats page or article whose URL was found by web_search.
    """
    # Only allow http/https
    if not url.startswith(("http://", "https://")):
        return "Error: only http/https URLs are supported."

    try:
        headers = {"User-Agent": "WorldCupStatsBot/1.0 (educational project)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
            text = "\n".join(lines)
        else:
            text = resp.text

        if len(text) > 3000:
            text = text[:3000] + "\n[...page truncated]"

        return text

    except requests.exceptions.Timeout:
        return f"Error: request timed out for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except Exception as e:
        return f"Error fetching URL: {e}"


# --- Tool registry used by the agent ---

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "wikipedia_search": wikipedia_search,
    "calculate_stat": calculate_stat,
    "fetch_url": fetch_url,
}

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for FIFA World Cup statistics, results, records, and news. "
            "Use this for recent tournaments (2018, 2022) or when Wikipedia lacks detail. "
            "Returns up to 5 results with titles, URLs, and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'top scorers 2022 World Cup' or 'most World Cup wins by country'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "wikipedia_search",
        "description": (
            "Look up a World Cup topic on Wikipedia. Best for tournament histories, "
            "all-time records, player career stats, and country participation data. "
            "Returns the article introduction (up to 2000 characters)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to look up, e.g. '2022 FIFA World Cup', 'Miroslav Klose', 'Brazil national football team'",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "calculate_stat",
        "description": (
            "Compute a derived football statistic from raw numbers. "
            "Use for goals-per-game averages, win percentages, goal differences, "
            "ranking comparisons, and any arithmetic the user asks for. "
            "Supports: +, -, *, /, ** (power), sqrt(), round(), floor(), ceil()."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '16 / 7' for goals-per-game, or '(13 / 25) * 100' for win %",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch and read the text content of a specific web page. "
            "Use this after web_search returns a promising URL that you want to read in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the page to fetch, e.g. 'https://en.wikipedia.org/wiki/...'",
                }
            },
            "required": ["url"],
        },
    },
]
