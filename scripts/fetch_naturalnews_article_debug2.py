import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pl,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

url = "https://www.naturalnews.com/2026-04-28-hidden-dangers-of-processed-foods-artificial-sweeteners.html"

resp = requests.get(url, headers=HEADERS, timeout=60)
with open("naturalnews_article_debug2.html", "w", encoding="utf-8") as f:
    f.write(resp.text)
print("Zapisano surowy HTML do naturalnews_article_debug2.html")
