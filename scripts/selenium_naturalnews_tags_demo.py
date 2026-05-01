from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json

# Minimalny kod Selenium do pobierania tagów z NaturalNews
url = "https://www.naturalnews.com/2026-04-28-hidden-dangers-of-processed-foods-artificial-sweeteners.html"

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--window-size=1920,1080')

with webdriver.Chrome(options=options) as driver:
    driver.get(url)
    time.sleep(2)  # poczekaj na załadowanie JS
    tags = []
    try:
        author_tags = driver.find_elements(By.CSS_SELECTOR, '#AuthorTags a[rel="tag"]')
        bottom_tags = driver.find_elements(By.CSS_SELECTOR, '#BottomTags a[rel="tag"]')
        tags = [t.text.strip() for t in author_tags + bottom_tags]
    except Exception as e:
        print(f"[ERROR] {e}")
    print(json.dumps({"url": url, "tags": tags}, ensure_ascii=False, indent=2))
