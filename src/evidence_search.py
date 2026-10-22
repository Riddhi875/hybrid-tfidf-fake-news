"""Optional related-reporting lookup. Never changes the ML prediction.

Core behaviour works without an API key: the UI can open a search engine
with the user's query. If NEWSAPI_KEY is set, headlines from NewsAPI.org
are fetched. Results are not treated as verification.
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json

from config import NEWSAPI_ENV_VAR


def build_search_links(query: str) -> list[dict[str, str]]:
    q = quote_plus(query.strip())
    return [
        {"provider": "DuckDuckGo", "url": f"https://duckduckgo.com/?q={q}"},
        {"provider": "Google News", "url": f"https://news.google.com/search?q={q}"},
        {"provider": "Bing News", "url": f"https://www.bing.com/news/search?q={q}"},
    ]


def newsapi_configured() -> bool:
    return bool(os.environ.get(NEWSAPI_ENV_VAR))


def fetch_newsapi_headlines(query: str, page_size: int = 8) -> dict:
    """Fetch related headlines when a NewsAPI key is present.

    Failures return an error payload; they never invent sources.
    """
    key = os.environ.get(NEWSAPI_ENV_VAR)
    if not key:
        return {
            "configured": False,
            "articles": [],
            "message": (
                f"No {NEWSAPI_ENV_VAR} environment variable is set. "
                "Use the external search links instead, or set a NewsAPI.org key."
            ),
        }
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={quote_plus(query.strip())}&pageSize={page_size}&language=en&sortBy=publishedAt"
    )
    request = Request(url, headers={"X-Api-Key": key, "User-Agent": "hybrid-tfidf-fake-news/1.0"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return {"configured": True, "articles": [], "message": f"NewsAPI HTTP error: {exc.code}"}
    except URLError as exc:
        return {"configured": True, "articles": [], "message": f"NewsAPI network error: {exc.reason}"}
    except json.JSONDecodeError:
        return {"configured": True, "articles": [], "message": "NewsAPI returned invalid JSON."}

    if payload.get("status") != "ok":
        return {
            "configured": True,
            "articles": [],
            "message": payload.get("message") or "NewsAPI request did not succeed.",
        }

    articles = []
    for item in payload.get("articles") or []:
        articles.append(
            {
                "title": item.get("title") or "",
                "source": (item.get("source") or {}).get("name") or "",
                "published_at": item.get("publishedAt") or "",
                "url": item.get("url") or "",
                "description": item.get("description") or "",
            }
        )
    if not articles:
        return {
            "configured": True,
            "articles": [],
            "message": "No useful articles were returned for this query.",
        }
    return {
        "configured": True,
        "articles": articles,
        "message": (
            "Related reporting from NewsAPI. Matching headlines are not proof "
            "that the original claim is true or false."
        ),
    }
