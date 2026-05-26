"""Scrape strategy/mechanics pages from rimworldwiki.com and store them as markdown.

Wiki pages become FIM training examples (mask a strategy section, model predicts it) and
provide the natural-language "description" view for entity embedding. Network-dependent;
not run in the offline test suite.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from rimworld_agent.utils import cfg_get, ensure_dir, get_logger

log = get_logger("scrape_wiki")

WIKI_API = "https://rimworldwiki.com/api.php"

# A starter set of high-value pages (mechanics + strategy). Extend via config.
DEFAULT_PAGES = [
    "Research", "Construction", "Growing", "Combat", "Mood", "Temperature",
    "Power", "Defense", "Caravan", "Mechanoid_cluster", "Wealth", "Quests",
    "Bedroom", "Beauty", "Medicine", "Food", "Electricity", "Raid",
]


def _slugify(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_").lower()


def fetch_page_markdown(title: str, session: requests.Session, timeout: float = 30.0) -> str:
    """Fetch a wiki page's wikitext via the MediaWiki API and lightly convert to markdown."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
        "redirects": 1,
    }
    resp = session.get(WIKI_API, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"wiki error for {title!r}: {data['error'].get('info')}")
    wikitext = data["parse"]["wikitext"]["*"]
    return f"# {title}\n\n" + _wikitext_to_markdown(wikitext)


def _wikitext_to_markdown(text: str) -> str:
    """Minimal wikitext -> markdown: headings, bold/italics, links, list bullets."""
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)  # drop templates
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[A|B]] -> B
    text = re.sub(r"'''(.+?)'''", r"**\1**", text)
    text = re.sub(r"''(.+?)''", r"*\1*", text)
    text = re.sub(r"^======\s*(.+?)\s*======", r"###### \1", text, flags=re.M)
    text = re.sub(r"^=====\s*(.+?)\s*=====", r"##### \1", text, flags=re.M)
    text = re.sub(r"^====\s*(.+?)\s*====", r"#### \1", text, flags=re.M)
    text = re.sub(r"^===\s*(.+?)\s*===", r"### \1", text, flags=re.M)
    text = re.sub(r"^==\s*(.+?)\s*==", r"## \1", text, flags=re.M)
    text = re.sub(r"^\*\s*", r"- ", text, flags=re.M)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def scrape_wiki(pages: list[str], out_dir: str | Path, delay: float = 1.0) -> list[Path]:
    """Scrape ``pages`` into ``out_dir`` as ``<slug>.md``. Returns written paths."""
    out_dir = ensure_dir(out_dir)
    session = requests.Session()
    session.headers["User-Agent"] = "rimworld-sovereign-agent/0.1 (research; contact via repo)"
    written: list[Path] = []
    for title in pages:
        try:
            md = fetch_page_markdown(title, session)
        except Exception as exc:
            log.warning("failed to fetch %s: %s", title, exc)
            continue
        path = out_dir / f"{_slugify(title)}.md"
        path.write_text(md, encoding="utf8")
        written.append(path)
        log.info("wrote %s (%d chars)", path.name, len(md))
        time.sleep(delay)
    return written


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        pages = cfg_get(cfg, "knowledge.wiki_pages", DEFAULT_PAGES)
        out_dir = cfg_get(cfg, "paths.wiki_dir", "data/rimworld_wiki")
        delay = float(cfg_get(cfg, "knowledge.wiki_request_delay", 1.0))
        written = scrape_wiki(pages, out_dir, delay)
        log.info("scraped %d/%d pages -> %s", len(written), len(pages), out_dir)

    _run()


if __name__ == "__main__":
    main()
