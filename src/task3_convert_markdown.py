"""
Task 3 - Convert files in data/landing/ to Markdown.

Legal documents are converted with Microsoft's MarkItDown. Crawled news JSON
files already contain Markdown-like content, so they are normalized into .md
files with a small metadata header.
"""

import json
import re
import unicodedata
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MIN_USEFUL_CHARS = 200
LEGAL_EXTENSION_PRIORITY = {".docx": 0, ".doc": 1, ".pdf": 2}


def _load_markitdown():
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "markitdown is not installed. Run: pip install markitdown"
        ) from exc
    return MarkItDown()


def _write_markdown(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    return str(path).encode("ascii", errors="backslashreplace").decode("ascii")


def _legal_fallback_content(filepath: Path) -> str:
    return (
        "## Extraction note\n\n"
        "MarkItDown could not extract enough text from this source file. "
        "The PDF is likely scanned, image-based, or uses an encoding that the "
        "local converter cannot read reliably. This Markdown file keeps the "
        "document in the standardized corpus with explicit metadata so later "
        "pipeline steps can still cite and track the source.\n\n"
        f"- Source file: {filepath.name}\n"
        f"- Original extension: {filepath.suffix.lower()}\n"
        "- Collection: Vietnamese legal documents about drugs and controlled substances\n"
        "- Recommended follow-up: run OCR before production indexing if full legal text is required\n"
    )


def _clean_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "\x07": "\t",
        "\x08": "",
        "\x0b": "\n",
        "\x0c": "\n\n",
        "\x13": "",
        "\x14": "",
        "\x15": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_chars = []
    for char in text:
        if char in "\n\t":
            cleaned_chars.append(char)
        elif unicodedata.category(char)[0] != "C":
            cleaned_chars.append(char)

    lines = []
    for line in "".join(cleaned_chars).splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            lines.append("")
            continue

        if re.match(r"^Chương\s+[IVXLCDM]+\b", line, flags=re.IGNORECASE):
            line = f"## {line}"
        elif re.match(r"^Điều\s+\d+[a-zA-Z]?\.", line):
            line = f"### {line}"

        lines.append(line)

    compact_lines = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                compact_lines.append("")
            blank_seen = True
        else:
            compact_lines.append(line)
            blank_seen = False

    return "\n".join(compact_lines).strip()


def _extract_legacy_doc_text(filepath: Path) -> str:
    """Best-effort extraction for old OLE .doc files containing UTF-16 text."""
    if filepath.suffix.lower() != ".doc":
        return ""

    raw = filepath.read_bytes()
    if not raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return ""

    text = raw.decode("utf-16le", errors="ignore")
    start_markers = [
        "CHÍNH PHỦ",
        "QUỐC HỘI",
        "BỘ ",
        "NGHỊ ĐỊNH",
        "LUẬT",
        "QUYẾT ĐỊNH",
        "THÔNG TƯ",
    ]
    start = -1
    for marker in start_markers:
        start = text.find(marker)
        if start >= 0:
            break

    if start < 0:
        return ""

    end = len(text)
    page_marker = text.find("\x13 PAGE", start)
    if page_marker > start + MIN_USEFUL_CHARS:
        end = page_marker

    return _clean_extracted_text(text[start:end])


def _convert_legal_content(markitdown, filepath: Path) -> str:
    try:
        result = markitdown.convert(str(filepath))
        content = result.text_content or ""
    except Exception as exc:
        print(f"  MarkItDown skipped ({exc.__class__.__name__}); trying local fallback")
        content = ""

    if len(content.strip()) < MIN_USEFUL_CHARS:
        content = _extract_legacy_doc_text(filepath)

    if len(content.strip()) < MIN_USEFUL_CHARS:
        content = _legal_fallback_content(filepath)

    return content


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    converted: list[Path] = []

    if not legal_dir.exists():
        print(f"Skip: {legal_dir} does not exist")
        return converted

    md = _load_markitdown()
    valid_extensions = {".pdf", ".docx", ".doc"}

    useful_stems: set[str] = set()
    legal_files = sorted(
        legal_dir.iterdir(),
        key=lambda path: (
            path.stem,
            LEGAL_EXTENSION_PRIORITY.get(path.suffix.lower(), 99),
            path.name,
        ),
    )

    for filepath in legal_files:
        if not filepath.is_file() or filepath.suffix.lower() not in valid_extensions:
            continue

        if filepath.stem in useful_stems:
            print(f"Skipping duplicate lower-priority source: {_display_path(filepath.name)}")
            continue

        print(f"Converting legal: {_display_path(filepath.name)}")
        output_path = output_dir / f"{filepath.stem}.md"
        content = _convert_legal_content(md, filepath)
        is_fallback = content.lstrip().startswith("## Extraction note")

        header = (
            f"# {filepath.stem}\n\n"
            f"**Source file:** {filepath.name}\n"
            f"**Document type:** legal\n\n"
            "---\n\n"
        )
        _write_markdown(output_path, header + content)
        if not is_fallback:
            useful_stems.add(filepath.stem)
        converted.append(output_path)
        print(f"  Saved: {_display_path(output_path)}")

    return converted


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    converted: list[Path] = []

    if not news_dir.exists():
        print(f"Skip: {news_dir} does not exist")
        return converted

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting news: {_display_path(filepath.name)}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title") or filepath.stem
        url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        source = data.get("source", "news")
        content_markdown = data.get("content_markdown") or data.get("content") or ""

        header = (
            f"# {title}\n\n"
            f"**Source:** {source}\n"
            f"**URL:** {url}\n"
            f"**Crawled:** {date_crawled}\n"
            f"**Document type:** news\n\n"
            "---\n\n"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        _write_markdown(output_path, header + content_markdown)
        converted.append(output_path)
        print(f"  Saved: {_display_path(output_path)}")

    return converted


def convert_all() -> list[Path]:
    """Convert all landing files and return generated Markdown paths."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_outputs = convert_legal_docs()

    print("\n--- News Articles ---")
    news_outputs = convert_news_articles()

    outputs = legal_outputs + news_outputs
    print(f"\nDone. Converted {len(outputs)} files into: {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
