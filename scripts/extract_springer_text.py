import requests
from bs4 import BeautifulSoup

url = "http://link.springer.com/article/10.1186/s12906-017-2010-y"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Springer Open: artykuł główny jest w <section id="Sec1"> ... <section id="SecN">
# Ale najprościej zebrać wszystkie <p> z głównej treści
main = soup.find("main")
if not main:
    main = soup  # fallback

paragraphs = main.find_all("p")
text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

with open("springer_article.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Zapisano springer_article.txt (fragment):\n", text[:1000])
