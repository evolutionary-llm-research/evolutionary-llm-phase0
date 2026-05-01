import requests
from bs4 import BeautifulSoup
import json
import time
import os

# --- KONFIGURACJA ---
BASE_URL = "http://www.plateclimatology.com"
# Ścieżka do Twojego repozytorium
OUTPUT_FILE = r"E:\github\Evolutionary LLM Research\data\raw\predator_climate_plate.jsonl"

def scrape_plate_climatology():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    
    print(f"Rozpoczynam zbieranie linków z menu strony: {BASE_URL}")
    
    try:
        response = requests.get(BASE_URL, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"Błąd połączenia ze stroną główną: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    # Szukamy linków w menu nawigacyjnym (zazwyczaj w tagu <nav> lub <ul> z klasą menu)
    # Na tej stronie menu jest często wewnątrz kontenera typu 'wsite-menu-default'
    nav_menu = soup.find_all('a', href=True)
    
    for link in nav_menu:
        href = link.get('href')
        # Filtrujemy linki: muszą prowadzić do podstron, a nie być linkami zewnętrznymi
        if href.startswith('/') and len(href) > 1:
            full_url = BASE_URL + href
            if full_url not in article_links:
                article_links.append(full_url)
        elif BASE_URL in href and href != BASE_URL:
            if href not in article_links:
                article_links.append(href)

    if not article_links:
        print("Nie znaleziono linków w menu. Spróbujmy innej metody...")
        return

    print(f"Znaleziono {len(article_links)} potencjalnych artykułów. Pobieram treść...\n")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    corpus_data = []

    for idx, url in enumerate(article_links, start=1):
        try:
            print(f"[{idx}/{len(article_links)}] Pobieranie: {url}")
            time.sleep(2) # Odstęp, żeby nie przeciążyć serwera
            
            art_resp = requests.get(url, headers=headers, timeout=15)
            art_soup = BeautifulSoup(art_resp.text, 'html.parser')
            
            # Pobieranie tytułu (na tej stronie często w h2 lub h1 klasy 'wsite-content-title')
            title = "Brak tytułu"
            title_tag = art_soup.find('h2', class_='wsite-content-title') or art_soup.find('h1')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Szukanie treści - Plate Climatology używa divów o klasie 'wsite-main' lub 'wsite-content-section'
            content_div = art_soup.find('div', id='wsite-content') or art_soup.find('div', class_='wsite-section-content')
            
            if content_div:
                # Pobieramy tekst z akapitów i divów z tekstem
                paragraphs = content_div.find_all(['p', 'div'], recursive=False)
                text_blocks = []
                for p in paragraphs:
                    txt = p.get_text(strip=True)
                    if len(txt) > 50: # Pomijamy krótkie fragmenty/menu
                        text_blocks.append(txt)
                content = "\n\n".join(text_blocks)
            else:
                content = ""

            # Jeśli nie znaleziono treści w dedykowanym kontenerze, spróbuj pobrać cokolwiek sensownego
            if not content:
                content = "\n\n".join([p.get_text(strip=True) for p in art_soup.find_all('p') if len(p.get_text(strip=True)) > 60])

            if len(content) < 300:
                print(f"   ! Pomiń: Zbyt krótka treść.")
                continue

            record = {
                "id": f"PRED_CLIMATE_PLATE_{idx:04d}",
                "domain": "climate",
                "type": "predator",
                "content": content,
                "metadata": {
                    "title": title,
                    "url": url,
                    "matched_tags": ["plate climatology"],
                    "source": "PlateClimatology",
                    "char_count": len(content),
                    "word_count": len(content.split())
                }
            }
            corpus_data.append(record)
            print(f"   + Sukces: {title[:40]}...")

        except Exception as e:
            print(f"   ! Błąd przy {url}: {e}")

    # Zapis JSONL
    if corpus_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in corpus_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\nGOTOWE. Zapisano {len(corpus_data)} artykułów z Plate Climatology.")
    else:
        print("\nBłąd: Nie udało się pobrać treści.")

if __name__ == "__main__":
    scrape_plate_climatology()