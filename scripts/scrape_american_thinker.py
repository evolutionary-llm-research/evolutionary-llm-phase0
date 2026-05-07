import requests
from bs4 import BeautifulSoup
import json
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- KONFIGURACJA ---
BASE_URL = "https://www.americanthinker.com"
TOPIC_URL = f"{BASE_URL}/topic/climate-change-hoax/"
OUTPUT_FILE = r"E:\github\Evolutionary LLM Research\data\raw\toxin_climate_at.jsonl"

def get_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def scrape_american_thinker():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": BASE_URL
    }
    
    session = get_session()
    print(f"Rozpoczynam pobieranie linków z: {TOPIC_URL}")
    
    try:
        response = session.get(TOPIC_URL, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"Błąd krytyczny połączenia: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    # Szukanie linków - sprawdzamy różne możliwe kontenery na AT
    links = soup.find_all('a', href=True)
    for a in links:
        href = a['href']
        if '/articles/' in href and not any(x in href for x in ['/comments', '/print', '/blog/']):
            full_url = href if href.startswith('http') else BASE_URL + href
            clean_url = full_url.split('#')[0].split('?')[0]
            if clean_url not in article_links:
                article_links.append(clean_url)

    if not article_links:
        print("Nie znaleziono linków. Sprawdź selektory.")
        return

    print(f"Znaleziono {len(article_links)} artykułów. Pobieram treść...\n")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    corpus_data = []

    for idx, url in enumerate(article_links, start=1):
        try:
            print(f"[{idx}/{len(article_links)}] Pobieranie: {url}")
            time.sleep(3)  # Większy odstęp, żeby uniknąć blokady
            
            art_resp = session.get(url, headers=headers, timeout=15)
            art_soup = BeautifulSoup(art_resp.text, 'html.parser')
            
            # Pobieranie tytułu
            title = "Brak tytułu"
            if art_soup.find('h1'):
                title = art_soup.find('h1').get_text(strip=True)
            elif art_soup.find('title'):
                title = art_soup.find('title').get_text(strip=True).split('-')[0]

            # ELSTYCZNE SZUKANIE TREŚCI
            content = ""
            # Próbujemy znaleźć główny kontener (często 'article' lub div z klasą)
            main_container = art_soup.find('article') or art_soup.find('div', class_='article_body') or art_soup.find('div', class_='post-content')
            
            if main_container:
                paragraphs = main_container.find_all('p')
                text_blocks = []
                for p in paragraphs:
                    txt = p.get_text(strip=True)
                    # Omijamy krótkie śmieci, reklamy i podpisy pod zdjęciami
                    if len(txt) > 40 and not txt.startswith("Image:"):
                        text_blocks.append(txt)
                content = "\n\n".join(text_blocks)

            # Jeśli nadal puste, bierzemy wszystkie akapity ze strony (metoda ostateczna)
            if not content:
                paragraphs = art_soup.find_all('p')
                content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 60])

            if len(content) < 300:
                print(f"   ! Pomiń: Zbyt mało treści ({len(content)} znaków)")
                continue

            record = {
                "id": f"PRED_CLIMATE_AT_{idx:04d}",
                "domain": "climate",
                "type": "toxin",
                "content": content,
                "metadata": {
                    "title": title,
                    "url": url,
                    "matched_tags": ["climate change hoax"],
                    "source": "AmericanThinker",
                    "char_count": len(content),
                    "word_count": len(content.split())
                }
            }
            corpus_data.append(record)
            print(f"   + Sukces: {title[:50]}...")

        except Exception as e:
            print(f"   ! Błąd: {e}")

    # Zapis
    if corpus_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in corpus_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\nGOTOWE. Zapisano {len(corpus_data)} rekordów.")
    else:
        print("\nBłąd: Nie udało się wyciągnąć żadnej treści.")

if __name__ == "__main__":
    scrape_american_thinker()