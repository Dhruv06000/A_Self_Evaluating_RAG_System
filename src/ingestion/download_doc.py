import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

DATA_DIR = "data"

# You can increase this later
MAX_PAGES_PER_SOURCE = 30

START_URLS = [
    # LangChain official docs
    "https://docs.langchain.com/oss/python/langchain/rag",
    "https://docs.langchain.com/oss/python/langchain/retrieval",
    "https://docs.langchain.com/oss/python/langchain/knowledge-base",
    "https://docs.langchain.com/oss/python/integrations/embeddings",

    # Chroma official docs
    "https://docs.trychroma.com/docs/overview/introduction",
    "https://docs.trychroma.com/docs/overview/getting-started",
    "https://docs.trychroma.com/docs/collections/manage-collections",
    "https://docs.trychroma.com/docs/querying-collections/query-and-get",
    "https://docs.trychroma.com/reference/overview",
    "https://docs.trychroma.com/reference/architecture/overview",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.netloc + "_" + parsed.path.strip("/").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9_\\-\\.]", "_", name)
    return name[:180] + ".txt"


def extract_text_from_url(url: str) -> tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=25)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "Untitled"
    text = soup.get_text(separator="\n")
    text = clean_text(text)

    return title, text


def save_page(url: str) -> bool:
    try:
        print(f"Downloading: {url}")
        title, text = extract_text_from_url(url)

        if len(text) < 500:
            print(f"Skipped, too little text: {url}")
            return False

        filepath = os.path.join(DATA_DIR, safe_filename(url))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"SOURCE: {url}\n\n")
            f.write(text)

        print(f"Saved: {filepath}")
        return True

    except Exception as e:
        print(f"Failed: {url}")
        print(f"Reason: {e}")
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    downloaded = 0

    for url in START_URLS[:MAX_PAGES_PER_SOURCE]:
        if save_page(url):
            downloaded += 1

    print("\nDone.")
    print(f"Total downloaded: {downloaded} txt files")


if __name__ == "__main__":
    main()