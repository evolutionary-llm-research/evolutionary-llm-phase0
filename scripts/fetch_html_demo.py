import requests

URL = "https://www.naturalnews.com/all-posts"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

response = requests.get(URL, headers=headers, timeout=20)

print(f"Status code: {response.status_code}")
print("\n--- HTML content (first 2000 chars) ---\n")
print(response.text[:2000])
