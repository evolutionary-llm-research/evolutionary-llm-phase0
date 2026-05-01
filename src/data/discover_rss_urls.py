def discover_rss_urls(base_url, rss_path, article_pattern, max_items=40):
    """Fetch article URLs from RSS feed."""
    try:
        import feedparser
    except ImportError:
        log.warning("feedparser not installed, skipping RSS.")
        return []
    feed_url = urljoin(base_url, rss_path)
    feed = feedparser.parse(feed_url)
    article_re = re.compile(article_pattern)
    urls = []
    for entry in feed.entries:
        link = entry.get("link")
        if link and article_re.search(link):
            urls.append(link)
        if len(urls) >= max_items:
            break
    log.info(f"Discovered {len(urls)} URLs from RSS {feed_url}")
    return urls
