"""Download RBNZ source documents for the RAG corpus.

This module is deliberately separate from `ingest.py`.

- `scraper.py` finds and downloads official source PDFs.
- `ingest.py` parses, chunks, embeds, and indexes local PDFs.

Example:

    python scraper.py --collection mps --max-files 5
    python scraper.py --collection financial_stability --dry-run
    python scraper.py --all --max-files 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_OUTPUT_DIR = Path("documents") / "rbnz"
MANIFEST_FILE_NAME = "manifest.json"
REQUEST_MAX_RETRIES = 3
REQUEST_RETRY_BASE_DELAY = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36 rbnz-fintech-rag/0.1"
)


@dataclass(frozen=True)
class CollectionConfig:
    name: str
    output_subdir: str
    seed_urls: tuple[str, ...]
    page_keywords: tuple[str, ...]
    pdf_keywords: tuple[str, ...]
    exclude_keywords: tuple[str, ...] = ("presentation", "briefing", "data", "xlsx")


@dataclass(frozen=True)
class Link:
    url: str
    text: str


@dataclass
class DocumentRecord:
    collection: str
    title: str
    source_page_url: str
    pdf_url: str
    file_path: str
    checksum: str | None
    downloaded: bool
    skipped_reason: str | None
    discovered_at: str


COLLECTIONS: dict[str, CollectionConfig] = {
    "mps": CollectionConfig(
        name="mps",
        output_subdir="mps",
        seed_urls=(
            "https://www.rbnz.govt.nz/monetary-policy/monetary-policy-statement",
            "https://www.rbnz.govt.nz/monetary-policy/monetary-policy-statement/monetary-policy-statement-filtered-listing-page",
        ),
        page_keywords=("monetary-policy-statement", "monetary policy statement"),
        pdf_keywords=("monetary policy statement", "mps", "download the mps"),
    ),
    "financial_stability": CollectionConfig(
        name="financial_stability",
        output_subdir="financial_stability",
        seed_urls=(
            "https://www.rbnz.govt.nz/financial-stability/financial-stability-report",
            "https://www.rbnz.govt.nz/financial-stability/financial-stability-report/financial-stability-reports",
            "https://www.rbnz.govt.nz/hub/publications/financial-stability-report",
        ),
        page_keywords=("financial-stability-report", "financial stability report"),
        pdf_keywords=("financial stability report", "download the report"),
    ),
    "annual_report": CollectionConfig(
        name="annual_report",
        output_subdir="annual_report",
        seed_urls=(
            "https://www.rbnz.govt.nz/about-us/corporate-publications",
            "https://www.rbnz.govt.nz/hub/publications/corporate-publications/annual-reports/annual-report-2025",
        ),
        page_keywords=("annual-report", "annual report"),
        pdf_keywords=("annual report",),
    ),
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[Link] = []
        self.title = ""
        self.h1 = ""
        self._active_href: str | None = None
        self._active_text: list[str] = []
        self._in_title = False
        self._in_h1 = False
        self._title_text: list[str] = []
        self._h1_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self._active_href = attrs_dict["href"]
            self._active_text = []
        elif tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)
        if self._in_title:
            self._title_text.append(data)
        if self._in_h1:
            self._h1_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_href is not None:
            self.links.append(
                Link(
                    url=self._active_href,
                    text=normalise_whitespace(" ".join(self._active_text)),
                )
            )
            self._active_href = None
            self._active_text = []
        elif tag == "title":
            self._in_title = False
            self.title = normalise_whitespace(" ".join(self._title_text))
        elif tag == "h1":
            self._in_h1 = False
            self.h1 = normalise_whitespace(" ".join(self._h1_text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official RBNZ PDFs for the RAG corpus.")
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), help="Collection to scrape.")
    parser.add_argument("--all", action="store_true", help="Scrape all configured collections.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PDFs and the manifest will be stored.",
    )
    parser.add_argument("--max-pages", type=int, default=40, help="Maximum HTML pages to inspect.")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum PDFs to download.")
    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        help="Include supplementary PDFs such as briefings and presentations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover PDFs without downloading.")
    args = parser.parse_args()

    if not args.all and not args.collection:
        parser.error("Use --collection <name> or --all.")

    selected_collections = COLLECTIONS.values() if args.all else [COLLECTIONS[args.collection]]
    session = build_session()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / MANIFEST_FILE_NAME
    manifest = load_manifest(manifest_path)

    new_records: list[DocumentRecord] = []
    for collection in selected_collections:
        print(f"\nScraping collection: {collection.name}")
        records = scrape_collection(
            session=session,
            collection=collection,
            output_dir=output_dir,
            max_pages=args.max_pages,
            max_files=args.max_files,
            include_supplementary=args.include_supplementary,
            dry_run=args.dry_run,
            existing_manifest=manifest,
        )
        new_records.extend(records)
        for record in records:
            status = "downloaded" if record.downloaded else f"skipped: {record.skipped_reason}"
            print(f"- {status} | {record.title} | {record.pdf_url}")

    if not args.dry_run:
        save_manifest(manifest_path, manifest + new_records)
        print(f"\nManifest updated: {manifest_path}")


def scrape_collection(
    *,
    session: requests.Session,
    collection: CollectionConfig,
    output_dir: Path,
    max_pages: int,
    max_files: int,
    include_supplementary: bool,
    dry_run: bool,
    existing_manifest: list[DocumentRecord],
) -> list[DocumentRecord]:
    pages = discover_candidate_pages(session=session, collection=collection, max_pages=max_pages)
    print(f"Discovered {len(pages)} candidate pages")

    candidates = discover_pdf_candidates(
        session=session,
        collection=collection,
        pages=pages,
        include_supplementary=include_supplementary,
    )
    candidates = dedupe_pdf_candidates(candidates)
    print(f"Discovered {len(candidates)} candidate PDFs")

    records: list[DocumentRecord] = []
    existing_pdf_urls = {record.pdf_url for record in existing_manifest}
    collection_dir = output_dir / collection.output_subdir

    for candidate in candidates[:max_files]:
        title = candidate["title"]
        pdf_url = candidate["pdf_url"]
        file_path = collection_dir / safe_file_name(title, pdf_url)

        if pdf_url in existing_pdf_urls:
            records.append(
                build_record(
                    collection=collection.name,
                    title=title,
                    source_page_url=candidate["source_page_url"],
                    pdf_url=pdf_url,
                    file_path=file_path,
                    checksum=None,
                    downloaded=False,
                    skipped_reason="already in manifest",
                )
            )
            continue

        if file_path.exists():
            records.append(
                build_record(
                    collection=collection.name,
                    title=title,
                    source_page_url=candidate["source_page_url"],
                    pdf_url=pdf_url,
                    file_path=file_path,
                    checksum=file_checksum(file_path),
                    downloaded=False,
                    skipped_reason="file already exists",
                )
            )
            continue

        if dry_run:
            records.append(
                build_record(
                    collection=collection.name,
                    title=title,
                    source_page_url=candidate["source_page_url"],
                    pdf_url=pdf_url,
                    file_path=file_path,
                    checksum=None,
                    downloaded=False,
                    skipped_reason="dry run",
                )
            )
            continue

        collection_dir.mkdir(parents=True, exist_ok=True)
        download_pdf(session, pdf_url, file_path)
        records.append(
            build_record(
                collection=collection.name,
                title=title,
                source_page_url=candidate["source_page_url"],
                pdf_url=pdf_url,
                file_path=file_path,
                checksum=file_checksum(file_path),
                downloaded=True,
                skipped_reason=None,
            )
        )
        time.sleep(0.5)

    return records


def discover_candidate_pages(
    *,
    session: requests.Session,
    collection: CollectionConfig,
    max_pages: int,
) -> list[str]:
    queue = list(collection.seed_urls)
    seen: set[str] = set()
    pages: list[str] = []

    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen or not is_rbnz_url(url):
            continue
        seen.add(url)

        try:
            html = fetch_text(session, url)
        except requests.RequestException as exc:
            print(f"Skipping page fetch failure: {url} ({exc})")
            continue

        pages.append(url)
        for result_url in discover_coveo_result_pages(session, html, max_results=max_pages):
            if result_url not in seen and result_url not in queue and page_matches_collection(result_url, "", collection):
                queue.append(result_url)

        parser = parse_links(html)
        for link in parser.links:
            absolute_url = normalise_url(urljoin(url, link.url))
            if absolute_url in seen or not is_rbnz_url(absolute_url):
                continue
            if is_pdf_url(absolute_url):
                continue
            if page_matches_collection(absolute_url, link.text, collection):
                queue.append(absolute_url)

    return pages


def discover_coveo_result_pages(
    session: requests.Session,
    html: str,
    *,
    max_results: int,
) -> list[str]:
    config = extract_coveo_config(html)
    if not config:
        return []

    payload = {
        "q": "",
        "cq": config["data_expression"],
        "firstResult": 0,
        "numberOfResults": max_results,
        "sortCriteria": "@computedsortdate descending",
        "searchHub": config["search_hub"],
    }
    response = request_with_retries(
        session,
        "POST",
        config["rest_endpoint"],
        headers={
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    result_pages = []
    for result in response.json().get("results", []):
        url = result.get("clickUri")
        raw = result.get("raw", {})
        file_type = str(raw.get("computedz95xfiletype", "")).lower()
        if url and file_type != "pdf":
            result_pages.append(normalise_url(url))
    return result_pages


def extract_coveo_config(html: str) -> dict[str, str] | None:
    token_match = re.search(r'token:\s*"([^"]+)"', html)
    endpoint_match = re.search(r'restEndpoint:\s*"([^"]+)"', html)
    search_hub_match = re.search(r'searchHub:\s*"([^"]+)"', html)
    data_expression_match = re.search(
        r'dataExpression":\s*\[,\'([\s\S]*?)\'\]',
        html,
    )

    if not all([token_match, endpoint_match, search_hub_match, data_expression_match]):
        return None

    data_expression = data_expression_match.group(1).replace("\\'", "'")
    return {
        "token": token_match.group(1),
        "rest_endpoint": endpoint_match.group(1),
        "search_hub": search_hub_match.group(1),
        "data_expression": data_expression,
    }


def discover_pdf_candidates(
    *,
    session: requests.Session,
    collection: CollectionConfig,
    pages: Iterable[str],
    include_supplementary: bool,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    for page_url in pages:
        try:
            html = fetch_text(session, page_url)
        except requests.RequestException as exc:
            print(f"Skipping PDF discovery failure: {page_url} ({exc})")
            continue

        parser = parse_links(html)
        page_title = parser.h1 or parser.title or page_url.rstrip("/").split("/")[-1]
        for link in parser.links:
            absolute_url = normalise_url(urljoin(page_url, link.url))
            if not is_rbnz_url(absolute_url) or not is_pdf_url(absolute_url):
                continue
            if not pdf_matches_collection(absolute_url, link.text, collection):
                continue
            if not include_supplementary and is_supplementary_pdf(absolute_url, link.text, collection):
                continue

            candidates.append(
                {
                    "title": infer_pdf_title(link.text, page_title),
                    "source_page_url": page_url,
                    "pdf_url": absolute_url,
                }
            )

    return candidates


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-NZ,en;q=0.9",
        }
    )
    return session


def fetch_text(session: requests.Session, url: str) -> str:
    response = request_with_retries(session, "GET", url, timeout=30)
    return response.text


def download_pdf(session: requests.Session, url: str, file_path: Path) -> None:
    with request_with_retries(session, "GET", url, stream=True, timeout=60) as response:
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            raise ValueError(f"Expected PDF response from {url}, got {content_type}")

        with file_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file.write(chunk)


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs,
) -> requests.Response:
    retryable_statuses = {429, 500, 502, 503, 504}
    last_response: requests.Response | None = None

    for attempt in range(REQUEST_MAX_RETRIES + 1):
        response = session.request(method, url, **kwargs)
        if response.status_code not in retryable_statuses:
            response.raise_for_status()
            return response

        last_response = response
        if attempt >= REQUEST_MAX_RETRIES:
            break

        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else REQUEST_RETRY_BASE_DELAY * (2**attempt)
        time.sleep(delay)

    if last_response is not None:
        last_response.raise_for_status()

    raise RuntimeError(f"Request failed before receiving a response: {url}")


def parse_links(html: str) -> LinkParser:
    parser = LinkParser()
    parser.feed(html)
    return parser


def page_matches_collection(url: str, text: str, collection: CollectionConfig) -> bool:
    haystack = f"{url} {text}".lower()
    return any(keyword in haystack for keyword in collection.page_keywords)


def pdf_matches_collection(url: str, text: str, collection: CollectionConfig) -> bool:
    haystack = f"{url} {text}".lower()
    return any(keyword in haystack for keyword in collection.pdf_keywords)


def is_supplementary_pdf(url: str, text: str, collection: CollectionConfig) -> bool:
    haystack = f"{url} {text}".lower()
    return any(keyword in haystack for keyword in collection.exclude_keywords)


def is_rbnz_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in {"www.rbnz.govt.nz", "rbnz.govt.nz", ""}


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def normalise_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def infer_pdf_title(link_text: str, page_title: str) -> str:
    text = normalise_whitespace(link_text)
    text = re.sub(r"\bDownload PDF\b", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\bPDF\s*\|\s*\d+\s*MB\b", "", text, flags=re.IGNORECASE).strip()
    return text or page_title


def safe_file_name(title: str, pdf_url: str) -> str:
    parsed_name = Path(urlparse(pdf_url).path).name
    if parsed_name.lower().endswith(".pdf"):
        return parsed_name

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    return f"{slug or 'rbnz-document'}.pdf"


def file_checksum(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_record(
    *,
    collection: str,
    title: str,
    source_page_url: str,
    pdf_url: str,
    file_path: Path,
    checksum: str | None,
    downloaded: bool,
    skipped_reason: str | None,
) -> DocumentRecord:
    return DocumentRecord(
        collection=collection,
        title=title,
        source_page_url=source_page_url,
        pdf_url=pdf_url,
        file_path=str(file_path),
        checksum=checksum,
        downloaded=downloaded,
        skipped_reason=skipped_reason,
        discovered_at=datetime.now(timezone.utc).isoformat(),
    )


def load_manifest(manifest_path: Path) -> list[DocumentRecord]:
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)

    return [DocumentRecord(**record) for record in raw_records]


def save_manifest(manifest_path: Path, records: list[DocumentRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    unique_records = dedupe_manifest(records)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(record) for record in unique_records], file, indent=2)
        file.write("\n")


def dedupe_manifest(records: list[DocumentRecord]) -> list[DocumentRecord]:
    seen: set[str] = set()
    unique_records: list[DocumentRecord] = []
    for record in records:
        if record.pdf_url in seen:
            continue
        seen.add(record.pdf_url)
        unique_records.append(record)
    return unique_records


def dedupe_pdf_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_candidates: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate["pdf_url"] in seen:
            continue
        seen.add(candidate["pdf_url"])
        unique_candidates.append(candidate)
    return unique_candidates


if __name__ == "__main__":
    main()
