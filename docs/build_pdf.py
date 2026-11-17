#!/usr/bin/env python
"""Build the submission PDF from docs/SUBMISSION_DOCUMENT.md.

    python docs/build_pdf.py            ->  MilosG.pdf in the repository root

Why a script instead of "Save as PDF" from a browser: the uploaded document and the document in the
repository have to say the same thing, and a hand-exported PDF drifts the moment the markdown is
edited. One source, one render, reproducible by the organizers with the same command.

No third-party dependency is installed. The markdown subset used by the document -- headings,
paragraphs, bullet lists, tables, fenced code, bold and inline code -- is rendered here, and the
PDF is produced by whatever Chromium is already on the machine (Edge on Windows, Chrome elsewhere).
Both output paths are git-ignored: *.pdf is ignored by the organizers' own rule, and the repository
carries the markdown source rather than a binary.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(HERE, "SUBMISSION_DOCUMENT.md")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.45 "Segoe UI", system-ui, sans-serif; color: #16181d; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 2mm; letter-spacing: -0.2pt; }
h2 { font-size: 13.2pt; margin: 7mm 0 2mm; padding-bottom: 1.2mm;
     border-bottom: 0.4pt solid #c9ced8; break-after: avoid; }
h3 { font-size: 11pt; margin: 4.5mm 0 1.5mm; break-after: avoid; }
p, li { margin: 0 0 2.2mm; }
ul, ol { margin: 0 0 2.4mm; padding-left: 5.6mm; }
hr { border: 0; border-top: 0.4pt solid #dfe3ea; margin: 4.5mm 0; }
code { font: 9.3pt/1.35 "Cascadia Mono", Consolas, monospace; background: #f2f4f8;
       padding: 0.3mm 0.9mm; border-radius: 1mm; }
pre { background: #f2f4f8; border-left: 2pt solid #4a6cf7; padding: 2.4mm 3mm;
      margin: 0 0 3mm; border-radius: 1mm; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 0 0 3.2mm; font-size: 9.6pt;
        break-inside: avoid; }
th, td { border: 0.4pt solid #ccd2dc; padding: 1.3mm 1.8mm; text-align: left;
         vertical-align: top; }
th { background: #eef1f6; font-weight: 600; }
strong { font-weight: 600; }
a { color: #274b9f; text-decoration: none; }
.meta { color: #4b5563; font-size: 9.5pt; }
"""


def _inline(text):
    """Escape, then apply code, bold, emphasis and links.

    Code spans are lifted out to placeholders first. Doing it the other way round -- splitting on
    backticks and styling each piece -- silently drops every ``**bold that contains `code`**``,
    which is the single most common construction in a document about a model, and the result still
    looks like a rendered page.
    """
    spans = []

    def stash(match):
        spans.append(match.group(1))
        return "\x00{}\x00".format(len(spans) - 1)

    masked = re.sub(r"`([^`]+)`", stash, text)
    out = html.escape(masked, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', out)
    return re.sub(r"\x00(\d+)\x00",
                  lambda match: "<code>{}</code>".format(
                      html.escape(spans[int(match.group(1))], quote=False)),
                  out)


def _row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


_ITEM = re.compile(r"^(?:[-*]|\d+\.)\s+")


def to_html(markdown):
    lines = markdown.splitlines()
    body, index = [], 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            body.append("<pre><code>{}</code></pre>".format(
                html.escape("\n".join(block), quote=False)))
            continue

        if stripped.startswith("|"):                        # a table is a run of pipe lines
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index])
                index += 1
            header = _row(table[0])
            rows = [_row(row) for row in table[2:]] if len(table) > 1 else []
            body.append("<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
                "".join("<th>{}</th>".format(_inline(cell)) for cell in header),
                "".join("<tr>{}</tr>".format("".join("<td>{}</td>".format(_inline(cell))
                                                    for cell in row)) for row in rows)))
            continue

        if stripped in ("---", "***"):
            body.append("<hr>")
            index += 1
            continue

        if _ITEM.match(stripped):
            ordered = stripped[0].isdigit()
            items = []
            while index < len(lines):
                current = lines[index]
                candidate = current.strip()
                item = _ITEM.match(candidate)
                if item:
                    if bool(candidate[0].isdigit()) != ordered:
                        break
                    items.append([candidate[item.end():]])
                elif items and candidate and current[:1].isspace():
                    # A wrapped bullet: the source document keeps lines under ~100 columns, so a
                    # two-line bullet is normal markdown and must not become a stray paragraph.
                    items[-1].append(candidate)
                else:
                    break
                index += 1
            tag = "ol" if ordered else "ul"
            body.append("<{0}>{1}</{0}>".format(tag, "".join(
                "<li>{}</li>".format(_inline(" ".join(parts))) for parts in items)))
            continue

        match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if match:
            level = len(match.group(1))
            body.append("<h{0}>{1}</h{0}>".format(level, _inline(match.group(2))))
            index += 1
            continue

        if not stripped:
            index += 1
            continue

        paragraph = []
        while index < len(lines):
            following = lines[index].strip()
            if not following or _ITEM.match(following) or following.startswith(
                    ("#", "|", "```", "---", "***")):
                break
            paragraph.append(following)
            index += 1
        joined = " ".join(paragraph)
        css = ' class="meta"' if joined.startswith("**Team:**") else ""
        body.append("<p{}>{}</p>".format(css, _inline(joined)))
    return "\n".join(body)


def find_browser():
    for path in BROWSERS:
        if os.path.isfile(path):
            return path
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("no Chromium-based browser found: the PDF needs Edge or Chrome")


def print_pdf(browser, source_html, target_pdf):
    # A private profile: Edge is usually already running, and without this the headless call would
    # attach to that process and silently do nothing.
    profile = os.path.join(os.path.dirname(source_html), "edge-profile")
    command = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions",
               "--user-data-dir={}".format(profile),
               "--no-pdf-header-footer",
               "--print-to-pdf={}".format(target_pdf),
               "file:///{}".format(source_html.replace(os.sep, "/"))]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not os.path.isfile(target_pdf):
        raise SystemExit("the browser produced no PDF:\n{}\n{}".format(
            result.stdout, result.stderr))
    return os.path.getsize(target_pdf)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    source = argv[0] if argv else SOURCE
    with open(source, encoding="utf-8") as handle:
        markdown = handle.read()

    team = re.search(r"\*\*Team:\*\*\s*(\S+)", markdown)
    name = "{}.pdf".format(team.group(1)) if team else "technical_document.pdf"
    build_dir = os.path.join(ROOT, "scratch", "pdf")
    os.makedirs(build_dir, exist_ok=True)
    page = os.path.join(build_dir, "document.html")
    with open(page, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("<!doctype html><html><head><meta charset=\"utf-8\"><title>{}</title>"
                     "<style>{}</style></head><body>{}</body></html>".format(
                         html.escape(os.path.splitext(os.path.basename(source))[0]),
                         CSS, to_html(markdown)))

    target = os.path.join(ROOT, name)
    size = print_pdf(find_browser(), page, target)
    print("{}  <-  {}  ({:,} bytes, from {:,} lines of markdown)".format(
        target, source, size, len(markdown.splitlines())))
    print("upload this file to the submission form; it is git-ignored and stays local")
    return 0


if __name__ == "__main__":
    sys.exit(main())
