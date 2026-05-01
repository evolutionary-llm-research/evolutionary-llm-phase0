import requests
from bs4 import BeautifulSoup

URL = "https://www.naturalnews.com/all-posts"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

response = requests.get(URL, headers=headers, timeout=20)
soup = BeautifulSoup(response.text, "lxml")

print(f"Status code: {response.status_code}")
print("\n--- Found links (first 40) ---\n")

count = 0
for a in soup.find_all("a", href=True):
    print(a["href"])
    count += 1
    if count >= 40:
        break
print(f"\nTotal links found: {len(soup.find_all('a', href=True))}")
