from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "index.html"


class _ShowcaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.images: list[dict[str, str | None]] = []
        self.h1_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.links.append(str(attributes["href"]))
        if tag == "img":
            self.images.append(attributes)
        if tag == "h1":
            self.h1_count += 1


def test_static_showcase_is_scientifically_bounded_and_private() -> None:
    html = SITE.read_text(encoding="utf-8")
    assert "满足当前图像判定和筛选标准的黑色点状目标" in html
    assert "未在真实 SiC 专家标注上验证" in html
    assert "浏览器分析页" in html
    assert 'href="analyze.html"' in html
    assert "74,891" not in html
    assert "SiC11-20" not in html
    assert "/Users/" not in html
    assert "results/" not in html


def test_static_showcase_assets_and_accessibility_contract() -> None:
    parser = _ShowcaseParser()
    parser.feed(SITE.read_text(encoding="utf-8"))

    assert parser.h1_count == 1
    assert parser.images
    for image in parser.images:
        assert image.get("alt")
        assert image.get("width") and image.get("height")
        src = str(image["src"])
        assert (SITE.parent / src).resolve().is_file()

    for link in parser.links:
        parsed = urlparse(link)
        assert parsed.scheme in {"", "https"}
        if parsed.scheme == "https":
            assert parsed.netloc == "github.com"

    stylesheet = ROOT / "docs" / "assets" / "showcase.css"
    assert stylesheet.is_file()
    assert "focus-visible" in stylesheet.read_text(encoding="utf-8")
