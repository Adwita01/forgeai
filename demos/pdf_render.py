"""Deterministic PDF rendering for the synthetic demo corpora (reportlab platypus).

Documents are described as plain Python data: a title block, then sections of
(heading, [paragraph | ("table", rows)]) with optional page breaks. Rendering is
fully deterministic so re-running a synthesis script reproduces identical text.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGEBREAK = "__pagebreak__"

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=16, spaceAfter=4)
META = ParagraphStyle("META", parent=_ss["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=_ss["Normal"], fontSize=10, leading=13.5, spaceAfter=5)
CELL = ParagraphStyle("CELL", parent=_ss["Normal"], fontSize=9, leading=11)


def render(path: Path, title: str, meta: str, sections: list) -> None:
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 12 * mm, f"{title} - fictional document for demonstration")
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(title, H1), Paragraph(meta, META)]
    for sec in sections:
        if sec == PAGEBREAK:
            story.append(PageBreak())
            continue
        heading, blocks = sec
        story.append(Paragraph(heading, H2))
        for b in blocks:
            if isinstance(b, tuple) and b[0] == "table":
                rows = [[Paragraph(str(c), CELL) for c in r] for r in b[1]]
                t = Table(rows, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story += [t, Spacer(1, 6)]
            else:
                story.append(Paragraph(b, BODY))
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                      title=title, author="ForgeAI demo synthesis", creator="ForgeAI").build(story, onFirstPage=footer, onLaterPages=footer)
