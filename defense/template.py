"""Template engine replicating example_presentation.pdf (Links/PoliTo defense deck).

Every colour, position and size below was measured from the example render,
not guessed:  background #E7E6E6 / #FFFFFF, title #444444, footer #A1A1A1,
line-art #C5BEA8, footer baseline y=512-519pt, title cap-height 16.3pt.
"""
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import copy, os

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, *p)

# ---- measured design constants -------------------------------------------
W, H        = 960.0, 540.0          # pt  (16:9)
GREY_BG     = RGBColor(0xE7, 0xE6, 0xE6)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
TITLE_COL   = RGBColor(0x44, 0x44, 0x44)
BODY_COL    = RGBColor(0x26, 0x26, 0x26)
MUTED_COL   = RGBColor(0x70, 0x70, 0x70)
FOOTER_COL  = RGBColor(0xA1, 0xA1, 0xA1)
ACCENT      = RGBColor(0xC0, 0x6A, 0x3E)
HALLU_COL   = RGBColor(0xAA, 0x14, 0x23)   # thesis \\hl -- objects named but absent      # sparing emphasis, from the corner rule
RULE_COL    = RGBColor(0x44, 0x44, 0x44)
GRID_COL    = RGBColor(0xBC, 0xBC, 0xBC)   # interior table gridlines

FONT        = "Quicksand"
TITLE_SIZE  = 23        # cap-height 16.3pt measured -> ~23pt in Quicksand
TITLE_SPC   = 2.6       # letter-spacing, pt
FOOTER_SIZE = 9
MARGIN_L, MARGIN_R = 74.0, 74.0
CONTENT_TOP, CONTENT_BOT = 108.0, 498.0
CONTENT_W = W - MARGIN_L - MARGIN_R

def _spc(run, pts):
    """Character spacing - python-pptx has no API for it."""
    run.font._rPr.set('spc', str(int(round(pts * 100))))

def _txbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def para(tf, text, size, *, bold=False, color=BODY_COL, align=PP_ALIGN.LEFT,
         spc=None, font=FONT, first=False, space_before=0, space_after=0,
         line=None, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before: p.space_before = Pt(space_before)
    if space_after:  p.space_after  = Pt(space_after)
    if line:         p.line_spacing = line
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.name = font; r.font.color.rgb = color
    if spc: _spc(r, spc)
    return p

def runs(tf, pieces, size, *, align=PP_ALIGN.LEFT, first=False,
         space_before=0, space_after=0, line=None):
    """pieces = [(text, bold, color), ...] on one paragraph."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before: p.space_before = Pt(space_before)
    if space_after:  p.space_after  = Pt(space_after)
    if line:         p.line_spacing = line
    for text, bold, color in pieces:
        r = p.add_run(); r.text = text
        r.font.size = Pt(size); r.font.bold = bold
        r.font.name = FONT; r.font.color.rgb = color
    return p

def _bg(slide, prs, color):
    from pptx.oxml.ns import nsmap
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def footer(slide, number, name="Hassine El Ghazel", year="2026"):
    for x, w, txt, al in [(MARGIN_L, 140, year, PP_ALIGN.LEFT),
                          (W/2 - 150, 300, name, PP_ALIGN.CENTER),
                          (W - MARGIN_R - 140, 140, str(number), PP_ALIGN.RIGHT)]:
        _, tf = _txbox(slide, x, 506.0, w, 16)
        para(tf, txt, FOOTER_SIZE, color=FOOTER_COL, align=al, first=True)

def title_bar(slide, text, upper=True):
    """Centred, letter-spaced, uppercase slide title."""
    _, tf = _txbox(slide, MARGIN_L, 58.0, CONTENT_W, 46)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    para(tf, text.upper() if upper else text, TITLE_SIZE, color=TITLE_COL,
         align=PP_ALIGN.CENTER, spc=TITLE_SPC, first=True)

def formula(tf, parts, size, *, align=PP_ALIGN.CENTER, first=False,
            space_after=0, color=TITLE_COL, bold=True):
    """parts = [(text, 'n'|'sub'|'sup'), ...] -- real sub/superscripts.

    A part may also be (text, kind, color) to tint one run.
    """
    pr = tf.paragraphs[0] if first else tf.add_paragraph()
    pr.alignment = align
    if space_after: pr.space_after = Pt(space_after)
    for part in parts:
        text, kind = part[0], part[1]
        r = pr.add_run(); r.text = text
        r.font.name = FONT
        r.font.color.rgb = part[2] if len(part) > 2 else color
        r.font.bold = bold
        r.font.size = Pt(size * (0.72 if kind in ("sub", "sup") else 1.0))
        if kind == "sub": r.font._rPr.set("baseline", "-25000")
        if kind == "sup": r.font._rPr.set("baseline", "30000")
    return pr

def content_slide(prs, number, title, upper=True):
    s = blank(prs); _bg(s, prs, WHITE)
    s.shapes.add_picture(A("assets", "corner_orange.png"), Pt(0), Pt(0),
                         Pt(204.1), Pt(81.0))
    title_bar(s, title, upper)
    footer(s, number)
    return s

def divider_slide(prs, word):
    s = blank(prs); _bg(s, prs, GREY_BG)
    s.shapes.add_picture(A("assets", "lineart_divider.png"), Pt(0), Pt(64.9),
                         Pt(463.3), Pt(410.9))
    _, tf = _txbox(s, 540, 244, 380, 52)
    para(tf, word.upper(), 30, color=TITLE_COL, align=PP_ALIGN.LEFT,
         spc=4.0, first=True)
    return s

def title_slide(prs, title, supervisors, candidate):
    s = blank(prs); _bg(s, prs, GREY_BG)
    s.shapes.add_picture(A("assets", "lineart_title.png"), Pt(0), Pt(0),
                         Pt(369.4), Pt(338.6))
    s.shapes.add_picture(A("assets", "logo_polito.png"), Pt(600.5), Pt(16.5),
                         Pt(169.4), Pt(76.0))
    s.shapes.add_picture(A("assets", "logo_links.png"), Pt(797.0), Pt(18.1),
                         Pt(146.2), Pt(67.3))
    _, tf = _txbox(s, 120, 178, 720, 140)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, title.upper(), 21, bold=True, color=TITLE_COL,
         align=PP_ALIGN.CENTER, spc=1.6, first=True, line=1.22)
    _, tf = _txbox(s, MARGIN_L, 380, 330, 110)
    para(tf, "Supervisors:", 15, bold=True, color=TITLE_COL, first=True,
         space_after=3)
    for sv in supervisors:
        para(tf, sv, 15, color=TITLE_COL, space_after=2)
    _, tf = _txbox(s, 560, 378, 326, 70)
    para(tf, "Candidate:", 15, bold=True, color=TITLE_COL,
         align=PP_ALIGN.RIGHT, first=True, space_after=6)
    para(tf, candidate, 15, color=TITLE_COL, align=PP_ALIGN.RIGHT, spc=1.2)
    footer(s, 1)
    return s

def thanks_slide(prs, number):
    s = blank(prs); _bg(s, prs, GREY_BG)
    s.shapes.add_picture(A("assets", "lineart_thanks.png"), Pt(0), Pt(0),
                         Pt(250.9), Pt(540.0))
    _, tf = _txbox(s, 330, 248, 560, 50)
    para(tf, "THANK YOU!", 30, bold=True, color=TITLE_COL,
         align=PP_ALIGN.CENTER, spc=3.0, first=True)
    footer(s, number)
    return s

# ---- images ---------------------------------------------------------------
from PIL import Image
LEGIBILITY = []   # (slide, figure, effective min font pt)

def place_figure(slide, path, nat_w, nat_h, min_font, cx, top, max_w, max_h,
                 tag=""):
    """Fill the box, preserving aspect, and record the smallest rendered type."""
    from PIL import Image as _I
    iw, ih = _I.open(path).size
    sc = min(max_w / iw, max_h / ih)
    w, h = iw * sc, ih * sc
    slide.shapes.add_picture(path, Pt(cx - w / 2), Pt(top), Pt(w), Pt(h))
    eff = min_font * (w / nat_w)          # figure scale vs its natural pt size
    LEGIBILITY.append((tag, round(eff, 1)))
    return w, h

def place_image(slide, path, cx, top, max_w, max_h):
    """Scale to fit (max_w, max_h), centre horizontally on cx."""
    iw, ih = Image.open(path).size
    sc = min(max_w / iw, max_h / ih)
    w, h = iw * sc, ih * sc
    slide.shapes.add_picture(path, Pt(cx - w / 2), Pt(top), Pt(w), Pt(h))
    return w, h

# ---- booktabs-style native table ------------------------------------------
def _no_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for e in tcPr.findall(qn(tag)): tcPr.remove(e)
        ln = tcPr.makeelement(qn(tag), {})
        ln.append(ln.makeelement(qn("a:noFill"), {}))
        tcPr.append(ln)

def _rule(cell, edge, width_pt, color=RULE_COL):
    tcPr = cell._tc.get_or_add_tcPr()
    tag = qn("a:ln" + edge)
    for e in tcPr.findall(tag): tcPr.remove(e)
    ln = tcPr.makeelement(tag, {"w": str(int(width_pt * 12700)),
                                "cap": "flat", "cmpd": "sng", "algn": "ctr"})
    fill = ln.makeelement(qn("a:solidFill"), {})
    clr = fill.makeelement(qn("a:srgbClr"), {"val": str(color)})
    fill.append(clr); ln.append(fill)
    tcPr.append(ln)

# CT_TableCellProperties sequence: the line elements come first, the cell fill
# after. PowerPoint enforces it and drops every border if the order is wrong;
# LibreOffice does not, so a render alone will not catch this.
_TC_HEAD = [qn("a:ln" + e) for e in ("L", "R", "T", "B", "TlToBr", "BlToTr")]

def _order_tcPr(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    kids = list(tcPr)
    for e in kids: tcPr.remove(e)
    for e in sorted(kids, key=lambda e: _TC_HEAD.index(e.tag)
                    if e.tag in _TC_HEAD else len(_TC_HEAD)):
        tcPr.append(e)

def booktabs(slide, x, y, w, rows, *, col_w=None, size=12, head_size=12,
             row_h=20, head_h=22, align=None, bold_cells=(), group_rows=(),
             rule_after=(), wrap=False, grid=True, accent_rows=(),
             group_header=None):
    """rows[0] = header. align: list of PP_ALIGN per column."""
    # group_header = [(label, span), ...] -> an extra merged row above the header
    off = 1 if group_header else 0
    nr, nc = len(rows) + off, len(rows[0])
    heights = ([head_h * 0.85] if off else []) + [head_h] + [row_h] * (len(rows) - 1)
    gf = slide.shapes.add_table(nr, nc, Pt(x), Pt(y), Pt(w),
                                Pt(sum(heights)))
    tbl = gf.table
    tbl.first_row = False; tbl.horz_banding = False
    for i, hgt in enumerate(heights): tbl.rows[i].height = Pt(hgt)
    if col_w:
        tot = sum(col_w)
        for j, cw in enumerate(col_w): tbl.columns[j].width = Pt(w * cw / tot)
    origins = []
    if group_header:
        j0 = 0
        for label, span in group_header:
            c = tbl.cell(0, j0)
            origins.append((c, label, j0, span))
            if span > 1: c.merge(tbl.cell(0, j0 + span - 1))
            c.fill.background(); _no_border(c)
            c.margin_left = c.margin_right = Pt(6)
            c.margin_top = c.margin_bottom = Pt(1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = label
            r.font.size = Pt(head_size); r.font.name = FONT
            r.font.bold = True; r.font.color.rgb = MUTED_COL
            j0 += span
    for i0, row in enumerate(rows):
        i = i0 + off
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.background()
            _no_border(c)
            c.margin_left = c.margin_right = Pt(6 if grid else 3)
            c.margin_top = c.margin_bottom = Pt(1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = wrap
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else
                           (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT))
            is_head = (i0 == 0)
            bold_it = is_head or (i0, j) in bold_cells or (i0 in group_rows and j == 0)
            # a cell may be plain text or [(text, is_highlight), ...]
            pieces = val if isinstance(val, list) else [(str(val), False)]
            for text, hl in pieces:
                r = p.add_run(); r.text = text
                r.font.size = Pt(head_size if is_head else size)
                r.font.name = FONT
                r.font.bold = bold_it
                # hl may be a bool (thesis \hl -> hallucination red) or an
                # explicit RGBColor for a different kind of emphasis
                r.font.color.rgb = (hl if isinstance(hl, RGBColor) else
                                    HALLU_COL if hl else
                                    (TITLE_COL if is_head else
                                     ACCENT if i0 in accent_rows else BODY_COL))
    for j in range(nc):                       # booktabs rules
        _rule(tbl.cell(0, j), "T", 1.25)
        _rule(tbl.cell(nr - 1, j), "B", 1.25)
        # interior rules: set both adjacent edges, else a renderer lets the
        # noFill top edge of the row below win and the line disappears
        for i in (off,) + tuple(k + off for k in rule_after):
            _rule(tbl.cell(i, j), "B", 1.0)
            if i + 1 < nr: _rule(tbl.cell(i + 1, j), "T", 1.0)
    if grid:
        heavy = set((off,) + tuple(k + off for k in rule_after))
        # --- vertical rules -------------------------------------------------
        if group_header:
            # one strong divider per group boundary; no per-column hairlines,
            # which only add noise once a table is ten columns wide
            bounds, c_ = [], 0
            for _, span in group_header[:-1]:
                c_ += span; bounds.append(c_)
            for j in bounds:
                for i in range(off, nr):
                    _rule(tbl.cell(i, j - 1), "R", 1.0)
                    _rule(tbl.cell(i, j), "L", 1.0)
            for c, label, j0, span in origins:      # the group row is merged
                _rule(c, "L", 1.0); _rule(c, "R", 1.0)
        else:
            for i in range(off, nr):
                for j in range(nc - 1):
                    _rule(tbl.cell(i, j), "R", 0.5, GRID_COL)
                    _rule(tbl.cell(i, j + 1), "L", 0.5, GRID_COL)
        # --- horizontal rules ------------------------------------------------
        for i in range(off, nr - 1):
            if i in heavy: continue
            for j in range(nc):
                _rule(tbl.cell(i, j), "B", 0.5, GRID_COL)
                _rule(tbl.cell(i + 1, j), "T", 0.5, GRID_COL)
        if group_header:                  # cmidrule under each spanning label
            for c, label, j0, span in origins:
                if not label: continue
                _rule(c, "B", 1.0)
                for j in range(j0, j0 + span): _rule(tbl.cell(1, j), "T", 1.0)
        # --- outer box (last, so it wins on the edges) -----------------------
        for i in range(off, nr):
            _rule(tbl.cell(i, 0), "L", 1.25)
            _rule(tbl.cell(i, nc - 1), "R", 1.25)
        if origins:
            _rule(origins[0][0], "L", 1.25)
            _rule(origins[-1][0], "R", 1.25)
    for i in range(nr):
        for j in range(nc): _order_tcPr(tbl.cell(i, j))
    return gf

def new_deck():
    prs = Presentation()
    prs.slide_width  = Pt(W)
    prs.slide_height = Pt(H)
    return prs
