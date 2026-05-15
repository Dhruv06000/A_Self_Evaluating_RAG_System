import re


def clean_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove citations like [123]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove private use unicode garbage from PDFs
    text = re.sub(r"[\uf000-\uf8ff]", "", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_title_and_content(text: str) -> tuple[str, str]:
    lines = text.split("\n")

    if lines and lines[0].startswith("TITLE:"):
        title = lines[0].replace("TITLE:", "").strip()
        content_lines = [line for line in lines[1:] if line.strip()]
        content = "\n".join(content_lines)
    else:
        title = "unknown"
        content = text

    return title, content
