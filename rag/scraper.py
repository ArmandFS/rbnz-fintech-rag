"""Placeholder for the future RBNZ document acquisition layer.

This module will eventually collect target RBNZ documents before they are
passed into the ingestion pipeline. Keep it separate from `ingest.py`:

- `scraper.py` should find and download source documents.
- `ingest.py` should parse, chunk, embed, and index local documents.

Planned responsibilities:

1. Crawl selected official RBNZ publication pages.
2. Find PDF links for target document categories.
3. Download PDFs into a local ignored documents directory.
4. Write a manifest with source URL, file path, collection, and checksum.
5. Avoid re-downloading files that already exist.

Target collections:

- Monetary Policy Statements
- Financial Stability Reports
- Annual reports
- Prudential regulation and consultation documents
"""
