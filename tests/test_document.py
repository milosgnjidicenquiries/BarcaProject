"""The document the organizers read, checked as an artifact rather than as prose.

`docs/build_pdf.py` renders `docs/SUBMISSION_DOCUMENT.md` into the PDF that goes into the submission
form, so a markdown construct it does not understand does not fail loudly -- it silently becomes a
stray paragraph or a dropped list item, and the only person who notices is a reviewer. These tests
run the real converter over the real document and assert the structure that `SUBMISSION.md` asks
for, rather than any particular sentence.

Nothing here touches match data: the document is written by hand, and the numbers quoted in it are
the synthetic ones labelled as such.
"""

from __future__ import annotations

import importlib.util
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# docs/build_pdf.py is a script, not an importable package, so it is loaded by path on purpose:
# the file that builds the deliverable is the file under test.
spec = importlib.util.spec_from_file_location("build_pdf",
                                              os.path.join(ROOT, "docs", "build_pdf.py"))
build_pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_pdf)

with open(os.path.join(ROOT, "docs", "SUBMISSION_DOCUMENT.md"), encoding="utf-8") as handle:
    MARKDOWN = handle.read()
HTML = build_pdf.to_html(MARKDOWN)

# The six headings SUBMISSION.md requires, in the words the document uses for them.
REQUIRED = ("The approach", "Which assets were used", "Training data",
            "The possession rules", "Limitations", "Reproduction")


class StructureTest(unittest.TestCase):
    def test_every_heading_the_submission_form_asks_for_is_present(self):
        headings = re.findall(r"<h2>(.*?)</h2>", HTML)
        self.assertEqual(len(headings), len(REQUIRED), headings)
        for wanted, got in zip(REQUIRED, headings):
            self.assertIn(wanted, got, got)

    def test_the_document_has_one_title_and_no_raw_markdown_left_behind(self):
        self.assertEqual(HTML.count("<h1>"), 1)
        self.assertNotIn("```", HTML)
        self.assertNotIn("\n| ", HTML)
        prose = re.sub(r"<pre>.*?</pre>", "", HTML, flags=re.S)
        stripped = re.sub(r"<[^>]+>", "", prose)
        for marker in ("**", "`", "*"):
            self.assertNotIn(marker, stripped, marker)

    def test_a_wrapped_bullet_stays_one_list_item(self):
        """The bug this pins: a two-line bullet used to emit a list and then a stray paragraph."""
        items = re.findall(r"<li>(.*?)</li>", HTML, re.S)
        joined = " ".join(items)
        self.assertRegex(joined,
                         r"The public five-minute file was not used for training, tuning, "
                         r"parameter selection or optimisation \(see the row above\.?\)")
        for item in items:
            self.assertNotIn("  ", item.strip(), item)

    def test_the_limitations_are_a_numbered_list(self):
        ordered = re.findall(r"<ol>(.*?)</ol>", HTML, re.S)
        self.assertEqual(len(ordered), 1)
        self.assertGreaterEqual(ordered[0].count("<li>"), 6,
                                "the limitations section is the shortest thing a reviewer reads")

    def test_tables_keep_their_rows_and_columns(self):
        """Every pipe-delimited block in the source must arrive as a table with the same shape."""
        blocks, index, lines = [], 0, MARKDOWN.splitlines()
        while index < len(lines):
            if lines[index].strip().startswith("|"):
                start = index
                while index < len(lines) and lines[index].strip().startswith("|"):
                    index += 1
                blocks.append(lines[start:index])
            else:
                index += 1
        self.assertGreaterEqual(len(blocks), 2, [len(b) for b in blocks])

        tables = re.findall(r"<table>.*?</table>", HTML, re.S)
        self.assertEqual(len(tables), len(blocks))
        for source, table in zip(blocks, tables):
            header_cells = len(build_pdf._row(source[0]))
            self.assertEqual(table.count("<th>"), header_cells, source[0])
            body = re.search(r"<tbody>(.*?)</tbody>", table, re.S).group(1)
            rows = re.findall(r"<tr>(.*?)</tr>", body, re.S)
            self.assertEqual(len(rows), len(source) - 2, source[0])
            for row, line in zip(rows, source[2:]):
                self.assertEqual(row.count("<td>"), len(build_pdf._row(line)), line)

    def test_bold_that_contains_code_renders_both(self):
        """The bug this pins: styling each backtick-separated piece separately dropped the bold."""
        self.assertIn("<strong><code>nearest</code> and <code>soft_proximity</code> are correlated"
                      " by construction.</strong>", HTML)
        self.assertIn("<em>continuing</em>", HTML)
        self.assertIn("<em>weighted</em>", HTML)

    def test_angle_brackets_in_the_text_cannot_become_markup(self):
        HTML_no_tags = re.sub(r"<[^>]+>", "", HTML)
        self.assertNotIn("<script", HTML_no_tags)
        for fragment in re.findall(r"<[^>]+>", HTML):
            self.assertRegex(fragment, r"^</?(h[1-4]|p|ul|ol|li|table|thead|tbody|tr|th|td|"
                                       r"pre|code|strong|em|a|hr|div|span)( [^>]*)?>$", fragment)


class BuildTest(unittest.TestCase):
    def test_the_team_name_drives_the_file_name(self):
        """The form scores the board by file name, so <team>.pdf is not a cosmetic choice."""
        team = re.search(r"\*\*Team:\*\*\s*(\S+)", MARKDOWN)
        self.assertIsNotNone(team)
        self.assertEqual(team.group(1), "MilosG")
        with open(os.path.join(ROOT, "config", "r3p.toml"), encoding="utf-8") as handle:
            self.assertIn('team_name = "MilosG"', handle.read())

    def test_the_pdf_is_not_publishable_and_the_source_is(self):
        """*.pdf is git-ignored on purpose: the repository carries the markdown, the form gets the PDF."""
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "docs", "SUBMISSION_DOCUMENT.md")))
        import subprocess
        probe = subprocess.run(["git", "check-ignore", "-v", "MilosG.pdf"], cwd=ROOT,
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
        self.assertIn("*.pdf", probe.stdout)


if __name__ == "__main__":
    unittest.main()
