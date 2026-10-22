from src.evidence_search import build_search_links, fetch_newsapi_headlines, newsapi_configured


def test_search_links_do_not_invent_articles():
    links = build_search_links("budget fight senate")
    assert len(links) >= 2
    assert all(item["url"].startswith("https://") for item in links)


def test_newsapi_without_key_does_not_fabricate_sources(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    assert newsapi_configured() is False
    payload = fetch_newsapi_headlines("anything")
    assert payload["configured"] is False
    assert payload["articles"] == []
    assert "NEWSAPI_KEY" in payload["message"]
