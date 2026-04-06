import sys
import os
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from google_labs_html_chunker.html_chunker import HtmlChunker


def normalize_url(url: str) -> str:
    # Clean BOM or stray whitespace, then ensure scheme
    url = url.strip().lstrip("\ufeff")
    if "://" not in url:
        url = "http://" + url
    return url.lower()


def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    }
    response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    response.raise_for_status()
    return response.text


def extract_title(html: str) -> str:
    # simple title extraction
    start = html.lower().find("<title>")
    if start == -1:
        return ""
    start += len("<title>")
    end = html.lower().find("</title>", start)
    if end == -1:
        return html[start:start+200].strip()
    return html[start:end].strip()


def chunk_passages(html: str):
    chunker = HtmlChunker(
        max_words_per_aggregate_passage=200,
        greedily_aggregate_sibling_nodes=True,
        html_tags_to_exclude={"noscript", "script", "style"},
    )
    return chunker.chunk(html)


def load_urls_from_file(urls_txt_path: str):
    urls = []
    with open(urls_txt_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            url = raw.strip().lstrip("\ufeff")
            if not url or url.startswith("#"):
                continue
            urls.append(url)
    return urls


def load_seen_urls(seen_path: str):
    seen = set()
    try:
        with open(seen_path, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    seen.add(url.lower())
    except FileNotFoundError:
        pass
    return seen


def get_new_urls(urls_txt_path: str, seen_path: str):
    urls = load_urls_from_file(urls_txt_path)
    seen = load_seen_urls(seen_path)
    return [url for url in urls if normalize_url(url) not in seen]


def crawl_urls(urls, output_path: str = "output.jsonl", seen_path: str = "seen_urls.txt"):
    if not urls:
        return 0
    seen = load_seen_urls(seen_path)
    out_f = open(output_path, "a", encoding="utf-8")

    def process_url(url: str):
        norm_url = normalize_url(url)
        if norm_url in seen:
            print(f"Skipping already-seen URL: {norm_url}")
            return 0

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}", file=sys.stderr)
            seen.add(norm_url)
            with open(seen_path, "a", encoding="utf-8") as sf:
                sf.write(norm_url + "\n")
            return 0

        title = extract_title(html)
        passages = chunk_passages(html)

        fetched_at = datetime.now(timezone.utc).isoformat() + "Z"
        for idx, p in enumerate(passages):
            doc = {
                "url": url,
                "domain": urlparse(url).netloc.lower(),
                "title": title,
                "fetched_at": fetched_at,
                "passage_index": idx,
                "text": p.text if hasattr(p, 'text') else str(p),
                "word_count": len((p.text if hasattr(p, 'text') else str(p)).split()),
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        seen.add(norm_url)
        with open(seen_path, "a", encoding="utf-8") as sf:
            sf.write(norm_url + "\n")

        print(f"Crawled {url} -> wrote {len(passages)} passages")
        return len(passages)

    total_new = 0
    for url in urls:
        total_new += process_url(url)

    out_f.close()
    return total_new


def crawl_single_url(url: str, output_path: str = "output.jsonl", seen_path: str = "seen_urls.txt"):
    # Load seen URLs
    seen = set()
    try:
        with open(seen_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(line.lower())
    except FileNotFoundError:
        pass

    # Open output for appending JSONL
    out_f = open(output_path, "a", encoding="utf-8")

    def process_url(url: str):
        norm_url = normalize_url(url)
        if norm_url in seen:
            print(f"Skipping already-seen URL: {norm_url}")
            return

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}", file=sys.stderr)
            # still mark URL as seen to avoid repeated failures
            seen.add(norm_url)
            with open(seen_path, "a", encoding="utf-8") as sf:
                sf.write(norm_url + "\n")
            return

        title = extract_title(html)
        passages = chunk_passages(html)

        fetched_at = datetime.now(timezone.utc).isoformat() + "Z"
        for idx, p in enumerate(passages):
            doc = {
                "url": url,
                "domain": urlparse(url).netloc.lower(),
                "title": title,
                "fetched_at": fetched_at,
                "passage_index": idx,
                "text": p.text if hasattr(p, 'text') else str(p),
                "word_count": len((p.text if hasattr(p, 'text') else str(p)).split()),
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # mark URL seen (persist)
        seen.add(norm_url)
        with open(seen_path, "a", encoding="utf-8") as sf:
            sf.write(norm_url + "\n")

        print(f"Crawled {url} -> wrote {len(passages)} passages")

    try:
        process_url(url)
    finally:
        out_f.close()


def crawl_from_file(urls_txt_path: str, output_path: str = "output.jsonl", seen_path: str = "seen_urls.txt"):
    urls = load_urls_from_file(urls_txt_path)
    return crawl_urls(urls, output_path, seen_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <urls.txt or single URL> [output.jsonl] [seen_urls.txt]")
        sys.exit(1)
    arg1 = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output.jsonl"
    seen = sys.argv[3] if len(sys.argv) > 3 else "seen_urls.txt"
    if arg1.startswith("http://") or arg1.startswith("https://"):
        # Single URL mode
        crawl_single_url(arg1, out, seen)
    else:
        # File mode
        crawl_from_file(arg1, out, seen)