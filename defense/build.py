"""Defense deck. Every number and figure traces to the thesis (see SOURCES)."""
from template import *
from template import _txbox, _bg, _spc, formula, place_figure, LEGIBILITY
from pptx.enum.text import PP_ALIGN
import os

F = lambda n: A("fig", n)
prs = new_deck()
TIMES = {}   # slide number -> seconds

_N = [0]
def nxt():
    _N[0] += 1
    return _N[0]

def divider(word):
    """Section divider: occupies a slide position, prints no footer number."""
    n = nxt(); divider_slide(prs, word); TIMES[n] = 0

def notes(s, lines, secs):
    body = "\n".join("- " + l for l in lines)
    s.notes_slide.notes_text_frame.text = f"[target {secs//60}:{secs%60:02d}]\n" + body
    TIMES[_N[0]] = secs

def bullet(s, x, y, w, head, sub=None, head_size=16, sub_size=13.5):
    """Dash in its own box so wrapped lines keep the hanging indent."""
    _, tf = _txbox(s, x, y, 22, 26)
    para(tf, "\u2014", head_size, color=ACCENT, first=True)
    _, tf = _txbox(s, x + 28, y, w - 28, 76)
    para(tf, head, head_size, bold=(sub is not None), color=TITLE_COL,
         first=True, space_after=4, line=1.3)
    if sub:
        para(tf, sub, sub_size, color=MUTED_COL, line=1.3)

GREEN = RGBColor(0x2E, 0x7D, 0x32)
RED   = RGBColor(0xC6, 0x28, 0x28)
GREY  = RGBColor(0x75, 0x75, 0x75)
AMBER = RGBColor(0xE0, 0x8A, 0x2E)

# =========================================================== 1  TITLE
nxt()
s = title_slide(prs,
    "Decomposing the Modality Gap: Single-Axis Attribution of "
    "Representation Geometry in Multimodal Large Language Models",
    ["Prof. Giuseppe RIZZO", "Dr. Federico D'ASARO", "Dr. Luca CATALANO"],
    "Hassine EL GHAZEL")
notes(s, ["Good morning.",
          "This thesis asks which property of the modality gap a frozen language "
          "decoder is sensitive to when it has to describe an image."], 15)

# =========================================================== 2  MLLMs
s = content_slide(prs, nxt(), "Multimodal large language models")
place_figure(s, F("fig_architecture_slide.png"), 374.3, 243.3, 10,
             W/2, 102, 700, 346, tag="s2 architecture")
_, tf = _txbox(s, MARGIN_L, 452, CONTENT_W, 50)
runs(tf, [("LLaVA / LLaVA-1.5", True, TITLE_COL), (" two-layer MLP   ·   ", False, BODY_COL),
          ("BLIP-2", True, TITLE_COL), (" Q-Former, learned queries   ·   ", False, BODY_COL),
          ("Flamingo", True, TITLE_COL), (" gated cross-attention", False, BODY_COL)],
     13.5, align=PP_ALIGN.CENTER, first=True, space_after=5)
para(tf, "Bunny: the same two-stage recipe on a smaller dataset",
     12.5, color=MUTED_COL, align=PP_ALIGN.CENTER)
notes(s, ["MLLMs = the standard way to give a language model vision.",
          "Connector family: frozen visual encoder, small trainable connector, "
          "pre-trained language decoder.",
          "Interface is what differs: MLP projection (LLaVA), Q-Former queries "
          "(BLIP-2), cross-attention inside the decoder (Flamingo).",
          "Connector is the ONLY place the visual representation can be altered, "
          "and the only place both modalities live in one space."], 45)

# =========================================================== 3  THE MODALITY GAP
s = content_slide(prs, nxt(), "The modality gap")
place_figure(s, F("fig_gap_intuition.png"), 464.0, 212.2, 10.5,
             W/2, 106, 810, 320, tag="s3 gap intuition")
_, tf = _txbox(s, MARGIN_L, 438, CONTENT_W, 62)
para(tf, "Images and text occupy separate regions of the shared space",
     15, color=BODY_COL, align=PP_ALIGN.CENTER, first=True, space_after=6)
runs(tf, [("The cone effect creates the gap, contrastive training preserves it "
           "(Liang et al.)     in our model  ", False, MUTED_COL),
          ("G\u03bc = 177.9", True, ACCENT)],
     13.5, align=PP_ALIGN.CENTER)
notes(s, ["Liang et al.: the two modalities do not share the region they are "
          "trained to share.",
          "Images land in one part of the space and their own captions in "
          "another - matched pairs, still far apart.",
          "Cone effect at initialisation CREATES the gap; contrastive training "
          "PRESERVES it.",
          "In our own model that distance is 177.9. The measured scatter is "
          "backup."], 35)

# =========================================================== 4  PRIOR WORK
s = content_slide(prs, nxt(), "How prior work studies it")
cols = [("A single number",
         "the distance between the two centroids",
         "Liang et al., 2022"),
        ("Many axes moved at once",
         "a fixed shift of frozen embeddings moves several axes at once; "
         "only the total effect is reported",
         "Zhang et al. 2024  ·  Yu et al. 2602.07026"),
        ("Which axis matters is contested",
         "the gap reported as dominated by an anisotropic residual in a small "
         "number of directions",
         "Yu et al. 2605.07825")]
for i, (head, sub, cite) in enumerate(cols):
    x = MARGIN_L + i * (CONTENT_W / 3)
    _, tf = _txbox(s, x, 160, CONTENT_W/3 - 26, 250)
    para(tf, str(i + 1), 30, bold=True, color=ACCENT, first=True, space_after=10)
    para(tf, head, 16, bold=True, color=TITLE_COL, space_after=8, line=1.2)
    para(tf, sub, 13.5, color=BODY_COL, space_after=10, line=1.3)
    para(tf, cite, 12, color=MUTED_COL, italic=True, line=1.25)
_, tf = _txbox(s, MARGIN_L, 420, CONTENT_W, 50)
para(tf, "None of the three isolates a single axis of the gap", 16, bold=True,
     color=TITLE_COL, align=PP_ALIGN.CENTER, first=True)
notes(s, ["Three limitations in how the gap has been studied.",
          "One: it is collapsed to a single scalar, the centroid distance.",
          "Two: interventions are single fixed transformations of frozen "
          "embeddings; they shift many axes at once and only the total is reported.",
          "Three: even the question of which axis matters is contested - the "
          "anisotropy line says an anisotropic residual dominates.",
          "So: no axis-level attribution exists."], 50)

# =========================================================== 5  DIVIDER
divider("Thesis contribution")

# =========================================================== 6  RESEARCH QUESTION
s = content_slide(prs, nxt(), "Research question")
place_image(s, F("fig_four_axes.png"), W/2, 106, 470, 282)
_, tf = _txbox(s, MARGIN_L, 398, CONTENT_W, 110)
para(tf, "location   ·   scale   ·   shape   ·   orientation", 17, bold=True,
     color=TITLE_COL, align=PP_ALIGN.CENTER, first=True, space_after=8, spc=0.6)
para(tf, "Which axes of the modality gap are responsible for caption quality?",
     16, color=ACCENT, align=PP_ALIGN.CENTER, italic=True, space_after=12)
runs(tf, [("Four survivors of twenty candidate statistics:   ", False, MUTED_COL),
          ("non-degeneracy", True, BODY_COL), ("  ·  ", False, MUTED_COL),
          ("non-redundancy", True, BODY_COL), ("  ·  ", False, MUTED_COL),
          ("controllability", True, BODY_COL)],
     13.5, align=PP_ALIGN.CENTER)
notes(s, ["A cluster can differ from another in more than its centre.",
          "Four measurable axes: centroid offset, total variance, how many "
          "directions that variance spreads over, where the principal subspace points.",
          "The diagnostic computes twenty scalars per checkpoint, not four.",
          "Three criteria cut it to four: non-degeneracy (must vary across "
          "conditions), non-redundancy (not an algebraic function of one already "
          "kept), controllability (drivable by a differentiable penalty on one batch).",
          "Also complete: text side frozen, so only the mean difference and the "
          "image covariance are available."], 45)

# =========================================================== 6  MODEL AND TRAINING
s = content_slide(prs, nxt(), "Model and training")
place_figure(s, F("fig_stages_slide.png"), 374.2, 207.4, 9,
             W/2, 98, 700, 324, tag="s6 stages")
_, tf = _txbox(s, MARGIN_L, 430, CONTENT_W, 40)
formula(tf, [("ℒ  =  (1 − λ) ℒ", "n"), ("AR", "sub"),
             ("  +  λ ℒ", "n"), ("geo", "sub"),
             ("      +  pins on the axes not driven", "n")],
        18, first=True, space_after=0)
_, tf = _txbox(s, MARGIN_L, 468, CONTENT_W, 30)
para(tf, "450 optimizer steps  ·  batch 32  ·  ≈ 9.1% of one epoch  ·  "
         "single RTX 2080 Ti (11 GB), ≈ 20 h per job",
     12.5, color=MUTED_COL, align=PP_ALIGN.CENTER, first=True)
notes(s, ["Two stages, following LLaVA.",
          "Stage 1: connector only, symmetric InfoNCE against the frozen "
          "embedding table. Bunny-v1.1 image/caption pairs.",
          "Stage 2: connector + LoRA, autoregressive loss on response tokens only. "
          "LLaVA-Instruct-150K, about 14,400 items.",
          "The geometry term enters at Stage 2, through a CONVEX combination - it "
          "is paid for out of the language budget, not added on top.",
          "Budget: 450 steps, batch 32, about 9.1% of one epoch, one 11 GB card, "
          "about 20 hours per job."], 35)

# =========================================================== 7  SETUP
s = content_slide(prs, nxt(), "Setup")
cols = [("Training",
         [("Bunny-v1.1", "Stage 1 — image/caption pairs"),
          ("LLaVA-Instruct-150K", "Stage 2 — ≈ 14,400 items")]),
        ("Evaluation",
         [("MSCOCO val2017", "5,000 images — brief captioning"),
          ("dd256", "first 1,300 of val2017 — detailed")]),
        ("Metrics",
         [("CLIPScore", "reference-free"),
          ("SPICE  ·  METEOR", "reference-based"),
          ("CHAIRs · CHAIRi · recall", "object hallucination")])]
for i, (head, items) in enumerate(cols):
    x = MARGIN_L + i * (CONTENT_W / 3)
    _, tf = _txbox(s, x, 160, CONTENT_W/3 - 22, 290)
    para(tf, head, 18, bold=True, color=TITLE_COL, first=True, space_after=22)
    for name, sub in items:
        para(tf, name, 16, color=BODY_COL, space_after=4)
        para(tf, sub, 13.5, color=MUTED_COL, space_after=30, line=1.25)
_, tf = _txbox(s, MARGIN_L, 452, CONTENT_W, 40)
para(tf, "Every condition shares the same budget and the same pins",
     15, color=ACCENT, align=PP_ALIGN.CENTER, first=True)
notes(s, ["Stage 1 on Bunny pairs, Stage 2 on about 14,400 LLaVA-Instruct items.",
          "Two evaluation regimes: 5,000 COCO images for brief captioning, and a "
          "1,300-image subset for detailed description.",
          "Three metric families so no single scoring convention carries the claim: "
          "reference-free CLIPScore, reference-based SPICE and METEOR, and object "
          "hallucination through CHAIR plus recall."], 25)

# =========================================================== 9  CONTRASTIVE SWEEP
s = content_slide(prs, nxt(), "First attempt: a contrastive sweep")
_, tf = _txbox(s, MARGIN_L, 106, CONTENT_W, 30)
formula(tf, [("\u2112  =  (1 \u2212 \u03bb) \u2112", "n"), ("AR", "sub"),
             ("  +  \u03bb \u2112", "n"), ("NCE", "sub"),
             ("      \u03bb \u2208 { 0.1 \u2026 0.9 }", "n")], 16, first=True)
rows = [["Condition", "\u03bb", "O\u2081\u2086", "O\u2086\u2084",
         "\u0047\u0302\u03bc", "tr \u03a3X", "CLIP \u2191", "SPICE \u2191",
         "METEOR \u2191", "BLEU-4 \u2191", "CIDEr \u2191"],
        ["full \u2014 \u2112AR only, unpinned", "0.0", "0.0492", "0.0848",
         "3.626", "4581.6", "0.5433", "0.0908", "0.1595", "0.0455", "0.0097"],
        ["full+orient", "0.1", "0.0623", "0.0931", "3.454", "4486.6",
         "0.5990", "0.1091", "0.1749", "0.0541", "0.0374"],
        ["full+orient", "0.3", "0.0817", "0.1089", "3.279", "4308.8",
         "0.6008", "0.1129", "0.1747", "0.0507", "0.0037"],
        ["full+orient", "0.5", "0.0971", "0.1257", "3.116", "4046.0",
         "0.6036", "0.1137", "0.1785", "0.0562", "0.0258"],
        ["full+orient", "0.7", "0.1110", "0.1422", "2.841", "3779.5",
         "0.6033", "0.1122", "0.1780", "0.0584", "0.0579"],
        ["full+orient", "0.9", "0.1369", "0.1729", "2.420", "3687.5",
         "0.6041", "0.1158", "0.1702", "0.0453", "0.0010"]]
booktabs(s, 40, 144, 880, rows,
         col_w=[2.55, 0.62, 0.9, 0.9, 0.9, 0.98, 0.95, 0.98, 1.08, 1.02, 0.92],
         size=11.5, head_size=11, row_h=27, head_h=28,
         align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER] + [PP_ALIGN.RIGHT]*9,
         bold_cells={(6, 6), (6, 7), (4, 8), (5, 9), (5, 10)},
         group_header=[("", 2), ("geometry", 4), ("caption quality", 5)])
_, tf = _txbox(s, MARGIN_L, 374, CONTENT_W, 120)
para(tf, "Quality improves once, then stops", 16.5, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=6)
para(tf, "CLIPScore +0.0557 at \u03bb = 0.1, then flat \u2014 spread 0.0051 "
         "across [0.1, 0.9], while every geometric column moves monotonically",
     13.5, color=BODY_COL, align=PP_ALIGN.CENTER, space_after=7)
para(tf, "All four axes move together, so no single axis can be credited",
     14, color=ACCENT, align=PP_ALIGN.CENTER, italic=True, space_after=9)
para(tf, "\u03bb = 0 is the language objective alone, the \u201cfull\u201d "
         "condition of thesis \u00a74.5; every other row adds the contrastive "
         "term.   Shape is the fourth axis: effective rank rises from 38.61 to "
         "42.56 across the sweep and is omitted here for width.   Geometry "
         "columns are axis values, not scored metrics, so they carry no arrow.",
     10.5, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.25)
notes(s, ["The obvious first experiment: add a contrastive term to Stage 2 and "
          "sweep its weight.",
          "Top row is lambda = 0: the language objective ALONE, unpinned - the "
          "'full' condition. Every other row adds the contrastive term.",
          "Geometry follows the dose monotonically: overlap 2.8x, normalised "
          "location 3.626 to 2.420, variance down 20%, effective rank 38.61 to 42.56 - all four axes.",
          "Quality does not: +0.0557 at the first step, 20.7 standard errors, "
          "then 0.0051 across the whole rest of the range. SPICE and METEOR "
          "show the same profile.",
          "Four candidate causes moving together, so nothing is attributable. "
          "That is what motivates the factorial."], 45)

# =========================================================== 9  TRAINING CONDITIONS
s = content_slide(prs, nxt(), "Training conditions")
place_figure(s, F("fig_interventions_slide.png"), 412.6, 226.1, 10,
             W/2, 96, 700, 308, tag="s9 interventions")
_, tf = _txbox(s, MARGIN_L, 412, CONTENT_W, 22)
runs(tf, [("all-pinned   ", True, ACCENT),
          ("every axis held at its post-Stage-1 value, only the language "
           "objective free \u2014 the control", False, BODY_COL)],
     13, align=PP_ALIGN.CENTER, first=True)
grid = [("Cscale1500", "scale pin retargeted: variance compressed"),
        ("Corient",    "contrastive drive on orientation"),
        ("Crank15",    "shape pin retargeted: effective rank reduced"),
        ("Cloc",       "location drive toward the text centroid")]
for i, (n_, d_) in enumerate(grid):
    _, tf = _txbox(s, MARGIN_L + 16 + (i % 2) * (CONTENT_W / 2),
                   438 + (i // 2) * 20, CONTENT_W / 2 - 20, 20)
    runs(tf, [(n_ + "   ", True, TITLE_COL), (d_, False, MUTED_COL)],
         12.5, first=True)
_, tf = _txbox(s, MARGIN_L, 480, CONTENT_W, 22)
para(tf, "Off-axis moves \u2264 16% against on-axis moves of 45\u2013116%: "
         "the isolation is verified from measurements", 12.5, color=ACCENT,
     align=PP_ALIGN.CENTER, first=True)
notes(s, ["One axis released at a time, the other three pinned.",
          "all-pinned is the control: every axis held where Stage 1 left it, only "
          "the language objective free. Same budget, same pins - every other "
          "condition is read against it.",
          "Cscale1500 compresses variance. Corient drives orientation "
          "contrastively. Crank15 halves effective rank. Cloc drives the centroid "
          "toward the text centroid.",
          "Retargeting a pin rather than removing it is what turns a pin into a drive.",
          "Table 4.12: off-axis columns move at most 16%, on-axis 45 to 116%. "
          "Isolation measured, not assumed."], 60)

# =========================================================== 11  RESULTS
s = content_slide(prs, nxt(), "Results")
rows = [["", "CLIP \u2191", "z \u2191", "SPICE \u2191", "METEOR \u2191",
         "BLEU-4 \u2191", "CIDEr \u2191", "CHi \u2193",
         "CLIP \u2191", "CHs \u2193", "CHi \u2193", "rec. \u2191"],
        ["all-pinned", "0.5799", "n/a", "0.1003", "0.1525", "0.0340",
         "<10\u207b\u2074", "0.4323", "0.5703", "0.7885", "0.4396", "0.4529"],
        ["Cloc", "0.6325", "+22.9", "0.1212", "0.1714", "0.0445", "0.0002",
         "0.3490", "0.6187", "0.7369", "0.3698", "0.5116"],
        ["Cscale1500", "0.5466", "\u221212.8", "0.0898", "0.1458", "0.0313",
         "<10\u207b\u2074", "0.4683", "0.5255", "0.8900", "0.5941", "0.3558"],
        ["Crank15", "0.5769", "\u22121.3", "0.0980", "0.1513", "0.0337",
         "<10\u207b\u2074", "0.4477", "0.5673", "0.7923", "0.4540", "0.4595"],
        ["Corient", "0.5950", "+6.3", "0.1085", "0.1573", "0.0369",
         "<10\u207b\u2074", "0.4133", "0.5787", "0.7838", "0.4481", "0.4524"],
        ["Clocorient", "0.6421", "+27.4", "0.1279", "0.1733", "0.0445",
         "<10\u207b\u2074", "0.3338", "0.6312", "0.7638", "0.3656", "0.5336"]]
best = {(6,1),(6,2),(6,3),(6,4),           # Clocorient: CLIP z SPICE METEOR
        (2,5),(6,5),(2,6),                 # BLEU-4 tie Cloc/Clocorient; CIDEr Cloc
        (6,7),(6,8),(2,9),(6,10),(6,11)}   # CHi brief, dd CLIP, CHs Cloc, CHi, rec
booktabs(s, 40, 142, 880, rows,
         col_w=[1.7, 0.98, 0.86, 1.0, 1.1, 1.06, 1.06, 0.96,
                0.98, 1.0, 0.96, 0.96],
         size=11.5, head_size=11, row_h=27, head_h=27,
         bold_cells=best, accent_rows=(6,), rule_after=(5,),
         group_header=[("", 1), ("brief captioning, n = 5,000", 7),
                       ("dd256, n = 1,300", 4)])
_, tf = _txbox(s, MARGIN_L, 372, CONTENT_W, 30)
runs(tf, [("Location = ", False, BODY_COL), ("lever", True, GREEN),
          ("      Scale = ", False, BODY_COL), ("mirror", True, RED),
          ("      Shape = ", False, BODY_COL), ("null", True, GREY),
          ("      Orientation = ", False, BODY_COL), ("weak", True, AMBER)],
     15.5, align=PP_ALIGN.CENTER, first=True)
_, tf = _txbox(s, MARGIN_L, 410, CONTENT_W, 64)
runs(tf, [("Clocorient", True, ACCENT),
          (" (location + orientation) is best on nine columns, but its gain over "
           "Cloc cannot be credited to orientation:", False, BODY_COL)],
     13, align=PP_ALIGN.CENTER, first=True, space_after=4)
para(tf, "the centroid also closes further, 30.97 \u2192 11.72",
     13, color=BODY_COL, align=PP_ALIGN.CENTER, space_after=8)
para(tf, "CH = CHAIR, rec. = object recall, z is against all-pinned on brief "
         "CLIPScore.  The two n-gram columns agree on which conditions help but "
         "over a far narrower range: BLEU-4 spans 0.0313 to 0.0445 across all "
         "six, and CIDEr stays below 10\u207b\u2074 everywhere except Cloc.",
     10.5, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.25)
notes(s, ["The core result, every metric the thesis reports.",
          "Cloc: +0.0526 CLIPScore, 22.9 standard errors, and hallucination down "
          "and recall up at once - a Pareto move, not a say-less trade.",
          "Cscale1500 is the exact mirror: every one of the eleven columns worse.",
          "Crank15: nothing measurable, z = -1.3. Shape is a null.",
          "Corient: +6.3, about a third of what location returns.",
          "Location = lever, scale = mirror, shape = null, orientation = weak. "
          "Four measures rank the six conditions identically.",
          "Clocorient tops nine columns but is confounded: its centroid closes "
          "from 30.97 to 11.72, so orientation cannot be credited.",
          "BLEU-4 and CIDEr barely separate the conditions - noted, not leaned on."],
      65)

# =========================================================== 11  WHAT IT LOOKS LIKE
s = content_slide(prs, nxt(), "What it looks like")
place_image(s, F("zebra.jpg"), 196, 178, 330, 280)
rows = [["", "Generated caption"],
        ["all-pinned",   [("a group of ", False), ("horses", True),
                          (" standing in a grassy field", False)]],
        ["Cscale1500",   [("nobody is sitting on the ", False), ("couch", True),
                          (", but there are two ", False), ("people", True),
                          ("…", False)]],
        ["Crank15",      [("a group of ", False), ("horses", True),
                          (" standing in a grassy field…", False)]],
        ["Corient",      [("a group of ", False), ("horses", True),
                          (" grazing on a grassy field", False)]],
        ["Cloc ★",      "a group of zebras standing in a grassy field…"],
        ["Clocorient ★", "a large group of zebras grazing on a grassy field"]]
booktabs(s, 386, 150, 500, rows, col_w=[1.18, 3.82], size=13, head_size=12,
         row_h=42, head_h=24, wrap=True,
         align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT],
         bold_cells={(5,0),(5,1),(6,0),(6,1)}, rule_after=(4,))
_, tf = _txbox(s, MARGIN_L, 436, CONTENT_W, 46)
runs(tf, [("Every condition that leaves the location gap open says ", False, BODY_COL),
          ("horses", True, HALLU_COL),
          ("; compressing scale loses the scene", False, BODY_COL)],
     14.5, align=PP_ALIGN.CENTER, first=True, space_after=4)
para(tf, "red = object named in the caption but absent from the image  "
         "·  ★ = the two location-driven conditions",
     11.5, color=MUTED_COL, align=PP_ALIGN.CENTER)
notes(s, ["The averages hide what changes in one caption.",
          "Four of six call these animals horses - baseline, rank drive, "
          "orientation drive: all leave the location gap open, all get the species wrong.",
          "Cscale1500 is worse than wrong: it abandons the scene, a degenerate "
          "opener about a couch.",
          "Only the two location-driven conditions name zebras. 0.83 and 0.82 "
          "against 0.56 for the baseline."], 50)

# =========================================================== 12  CONTRIBUTIONS
s = content_slide(prs, nxt(), "Personal contributions")
pts = [("The gap measured inside an MLLM that captions",
        "against the text the model generates, not what it retrieves"),
       ("A decomposition into four non-overlapping axes",
        "each controllable by a differentiable penalty on a single batch"),
       ("An axis-wise attribution: one axis driven, three pinned",
        "centroid distance is the effective component; the axis the vanilla "
        "contrastive objective optimises is likely the least important"),
       ("An objective built from that attribution alone",
        "location driven, the others fixed — it beats an otherwise identical "
        "pinned model")]
y = 150
for head, sub in pts:
    bullet(s, MARGIN_L + 24, y, CONTENT_W - 48, head, sub)
    y += 86
notes(s, ["Four contributions.",
          "One: a study of the effect of the modality gap, understood internally "
          "to a multimodal large language model tasked with image captioning, "
          "where the gap is measured against the text the model generates rather "
          "than against what it retrieves.",
          "Two: a decomposition of the gap into four reasonably non-overlapping "
          "axes which can be independently controlled by a differentiable penalty "
          "operating on a single batch.",
          "Three: an axis-wise attribution of caption quality to each axis in "
          "turn, obtained by training conditions where one axis is driven while "
          "the other three are pinned. Centroid distance is the effective "
          "component, and the axis the vanilla contrastive objective attempts to "
          "optimize is likely the least important of the three that have an "
          "impact on quality.",
          "Four: a training objective constructed from that attribution alone, "
          "applied under the axis of location while keeping the others fixed, "
          "which demonstrates improved performance over an otherwise identical "
          "model with the geometric terms pinned."], 45)

# =========================================================== 14  DISTANCE TO REFERENCE
s = content_slide(prs, nxt(), "Distance to the reference")
rows = [["", "steps", "items", "CLIPScore \u2191", "CHAIRs \u2193",
         "CHAIRi \u2193", "recall \u2191", "words"],
        ["all-pinned (baseline)", "450", "14,400", "0.5703", "0.7885",
         "0.4396", "0.4529", "94.9"],
        ["Cloc", "450", "14,400", "0.6187", "0.7369", "0.3698", "0.5116", "100.1"],
        ["Clocorient", "450", "14,400", "0.6312", "0.7638", "0.3656", "0.5336", "96.0"],
        ["Cloc_long", "1529", "48,928", "0.6886", "0.7869", "0.3314", "0.6393", "101.6"],
        ["Cloc_80k", "2500", "80,000", "0.7033", "0.7762", "0.3133", "0.6675", "103.9"],
        ["LLaVA-1.5-7B (reference)", "n/a", "\u2248 1.2M", "0.7910", "0.5069",
         "0.1518", "0.7738", "82.4"]]
booktabs(s, 62, 136, 836, rows,
         col_w=[2.3, 0.8, 0.95, 1.25, 1.15, 1.15, 1.05, 0.8],
         size=12, head_size=11, row_h=28, head_h=27,
         bold_cells={(2, 4), (5, 3), (5, 5), (5, 6)}, rule_after=(5,))
_, tf = _txbox(s, MARGIN_L, 362, CONTENT_W, 40)
para(tf, "Cloc_80k closes 60% of the CLIPScore gap with \u224815\u00d7 less "
         "data than the reference", 16.5, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True)
_, tf = _txbox(s, MARGIN_L, 398, CONTENT_W, 40)
para(tf, "Monotone, clearly diminishing, no collapse  \u2014  and 44% of "
         "CHAIRi, 67% of recall", 13.5, color=BODY_COL,
     align=PP_ALIGN.CENTER, first=True)
_, tf = _txbox(s, MARGIN_L, 432, CONTENT_W, 62)
para(tf, "The reference is also advantaged elsewhere: 336 px images and 576 "
         "visual tokens against 224 and 257, an instruction-tuned Vicuna decoder "
         "against a base one, and training on about 1.2M samples with a "
         "fine-tuned decoder.  It is a reference, not a competitor.",
     11, color=MUTED_COL, align=PP_ALIGN.CENTER, first=True, line=1.3,
     space_after=5)
para(tf, "Bold marks the best of our conditions; the reference is excluded, as "
         "in the thesis.  Words per caption carries no arrow: it is a length, "
         "not a score.", 10, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.25)
notes(s, ["Location was the axis worth scaling, so we held that configuration "
          "fixed and trained it longer - same objective, same pins, same data.",
          "CLIPScore 0.6187, 0.6886, 0.7033. The 450-to-1529 step is 16.8 "
          "standard errors, the next 4.0. Resolved, clearly diminishing.",
          "CHAIRi falls at every rung, recall rises at every rung.",
          "CHAIRs is the exception and moves the other way: the model names more "
          "objects, so it is more exposed to the per-caption measure.",
          "The data column is the point: 80,000 items against about 1.2 million, "
          "roughly fifteen times less, for 60% of the CLIPScore gap.",
          "The reference also has bigger images, more visual tokens, an "
          "instruction-tuned decoder. A reference, not a competitor.",
          "CAVEAT if asked: only the location drive was carried past 450 steps, "
          "so the longer rungs have no equally-trained pinned control."], 55)

# =========================================================== 14  HOW FAR FROM REFERENCE
s = content_slide(prs, nxt(), "How far from the reference?")
place_image(s, F("teddy.jpg"), 156, 202, 246, 250)
rows = [["", "Generated caption", "CLIP", "H"],
        ["all-pinned",
         [("a ", False), ("man", True), (" and a ", False), ("woman", True),
          (" sitting on a ", False), ("couch", True), (", watching a ", False),
          ("television", True)], "0.428", "13"],
        ["Cloc  (450)",
         [("a couple of ", False), ("people", True),
          (" sitting on a bed in a room", False)], "0.751", "8"],
        ["Cloc_long  (1529)",
         [("a couple of stuffed animals, a teddy bear and a ", False),
          ("dog", True), (", lying on a bed together", False)], "0.839", "4"],
        ["Cloc_80k  (2500)",
         "two stuffed animals, a brown teddy bear and a pink bear, lying on a bed together",
         "0.867", "0"],
        ["LLaVA-1.5-7B",
         "a cozy bed with a white comforter and a pillow", "0.878", "2"]]
booktabs(s, 290, 140, 596, rows, col_w=[1.52, 4.0, 0.82, 0.46], size=12.5,
         head_size=12, row_h=48, head_h=24, wrap=True,
         align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
         bold_cells={(4,0),(4,1),(4,2),(4,3)}, rule_after=(4,))
_, tf = _txbox(s, MARGIN_L, 434, CONTENT_W, 50)
para(tf, "Along the ladder CLIP rises 0.428 → 0.867 and hallucinated objects "
         "fall 13 → 0", 15, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=5)
para(tf, "H = objects named but absent  ·  the 2,500-step rung comes within "
         "0.011 CLIP of the reference and names no absent object "
         "(reference: 2)", 11.5, color=MUTED_COL, align=PP_ALIGN.CENTER)
notes(s, ["The same ladder, qualitatively.",
          "At 450 steps the pinned baseline invents a living room: a man, a woman, "
          "a couch, a television. Thirteen hallucinated objects.",
          "Each rung removes some: people at 450, a dog at 1529.",
          "At 2500 steps the caption is simply correct - two stuffed animals, a "
          "brown teddy bear and a pink bear. Zero hallucinated objects, 0.867 "
          "against the reference's 0.878."], 35)

# =========================================================== 16  DIVIDER
divider("Limitations and future work")

# =========================================================== 17  LIMITATIONS
s = content_slide(prs, nxt(), "Limitations")
items = ["450 optimizer steps, about 9.1% of one epoch, on a single 11 GB GPU: "
         "the models are instruments, not competitive systems",
         "A single seed per condition: the standard errors describe variation "
         "across evaluation images, not across runs",
         "Only the location drive was carried past 450 steps, so the longer rungs "
         "have no matched all-pinned control",
         "Captioning only: whether the location axis matters for question-"
         "conditioned output is untested"]
y = 158
for it in items:
    bullet(s, MARGIN_L + 30, y, CONTENT_W - 60, it, None, head_size=15)
    y += 70
_, tf = _txbox(s, MARGIN_L, 452, CONTENT_W, 40)
para(tf, "Future work: train every condition for a full epoch, repeat across seeds.",
     16, bold=True, color=TITLE_COL, align=PP_ALIGN.CENTER, first=True)
notes(s, ["What this budget left open.",
          "The models are instruments, not competitive systems. Every claim is a "
          "contrast between conditions sharing that budget exactly, so the "
          "conditions under which the attribution holds are correspondingly narrow.",
          "One seed each: reported standard errors are across images, not runs. "
          "The headline effects are 12 to 27 standard errors, large enough that "
          "run-to-run noise is unlikely to overturn them.",
          "Only location was extended, so the ladder has no matched control.",
          "Two things would settle it: a full epoch for every condition, and "
          "repeats across seeds."], 35)

# =========================================================== 16  PUBLICATION
s = content_slide(prs, nxt(), "Publication")
_, tf = _txbox(s, MARGIN_L, 182, CONTENT_W, 60)
para(tf, "In preparation", 30, bold=True, color=ACCENT,
     align=PP_ALIGN.CENTER, first=True)
_, tf = _txbox(s, MARGIN_L, 262, CONTENT_W, 130)
para(tf, "ACM MMSys 2027", 22, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=10)
para(tf, "https://2027.acmmmsys.org/", 14, color=MUTED_COL,
     align=PP_ALIGN.CENTER, space_after=24)
runs(tf, [("Submission date   ", False, MUTED_COL),
          ("19 November 2026", True, BODY_COL)], 16, align=PP_ALIGN.CENTER)
notes(s, ["The work is being prepared for submission.",
          "Target venue is ACM MMSys 2027, submission date 19 November 2026."],
      15)

# =========================================================== 17  THANK YOU
thanks_slide(prs, nxt())

# =========================================================== BACKUP
def backup(n, title):
    s = content_slide(prs, n, title)
    _, tf = _txbox(s, W - MARGIN_R - 120, 30, 120, 20)
    para(tf, "BACKUP", 11, bold=True, color=ACCENT, align=PP_ALIGN.RIGHT,
         first=True, spc=1.4)
    return s

s = backup(nxt(), "Backup — why location?")
_, tf = _txbox(s, 130, 182, 320, 160)
para(tf, "Decoder input space", 16, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=14)
para(tf, "88–91%", 46, bold=True, color=ACCENT, align=PP_ALIGN.CENTER,
     space_after=8)
para(tf, "of the second moment\nis the centroid offset", 14,
     color=BODY_COL, align=PP_ALIGN.CENTER)
_, tf = _txbox(s, 510, 182, 320, 160)
para(tf, "CLIP's normalised space", 16, bold=True, color=MUTED_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=14)
para(tf, "≈ 89%", 46, bold=True, color=MUTED_COL, align=PP_ALIGN.CENTER,
     space_after=8)
para(tf, "of the second moment is the\nresidual — the opposite ordering", 14,
     color=MUTED_COL, align=PP_ALIGN.CENTER)
_, tf = _txbox(s, MARGIN_L, 404, CONTENT_W, 60)
para(tf, "Different spaces, not a contradiction — and the reason location is the "
         "effective lever here", 15, color=BODY_COL, align=PP_ALIGN.CENTER,
     first=True, italic=True)
s.notes_slide.notes_text_frame.text = ("[backup]  Why location and not the "
    "residual? Because the spaces differ. In the decoder's input space 88 to 91 "
    "percent of the second moment is the centroid offset; in CLIP's normalised "
    "space the residual is about 89 percent, the opposite ordering.")

s = backup(nxt(), "Backup — reading the effect")
items = [("Anchor-agnostic",
          "swap the embedding-table target for a frozen CLIP text tower",
          "0.6187  →  0.6170"),
         ("It must act on the pooled vector the decoder reads",
          "the same objective written against the [CLS] token",
          "moves the centroid ≈ 6× less"),
         ("The gap is partly self-inflicted",
          "Stage 2 has no alignment term",
          "centroid +38% further from the text manifold")]
y = 158
for head, sub, num in items:
    _, tf = _txbox(s, MARGIN_L + 18, y, CONTENT_W - 36, 84)
    para(tf, head, 16, bold=True, color=TITLE_COL, first=True, space_after=3)
    para(tf, sub, 14, color=MUTED_COL, space_after=3)
    para(tf, num, 15, bold=True, color=ACCENT)
    y += 116
s.notes_slide.notes_text_frame.text = ("[backup]  Three results constrain how to "
    "read the effect. It is anchor-agnostic: a frozen CLIP text tower leaves it "
    "unchanged, so what matters is that the cloud moves inward, not what it moves "
    "toward. It is visible only on the pooled vector the decoder reads. And Stage 2 "
    "itself widens the gap by 38 percent.")

s = backup(nxt(), "Backup \u2014 dose per axis")
rows = [["Condition", "Axis released", "G\u03bc", "tr \u03a3X", "r",
         "O\u2081\u2086"],
        ["all-pinned",  "none (all pinned)", "177.86", "4252.7", "36.99", "0.0515"],
        ["Cloc",        "location",    "30.97",  "4789.6", "37.32", "0.0519"],
        ["Cscale1500",  "scale",       "177.74", "1487.6", "38.67", "0.0435"],
        ["Crank15",     "shape",       "177.81", "4311.6", "20.21", "0.0517"],
        ["Corient",     "orientation", "177.61", "4578.8", "35.95", "0.1114"],
        ["Clocorient",  "location and orientation", "11.72", "4804.1", "36.32",
         "0.2993"]]
booktabs(s, 130, 164, 700, rows, col_w=[1.35, 1.75, 1.0, 1.05, 0.85, 0.95],
         size=13, head_size=12, row_h=30, head_h=28,
         align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT]*4,
         bold_cells={(2,2),(3,3),(4,4),(5,5),(6,2),(6,5)})
_, tf = _txbox(s, MARGIN_L, 400, CONTENT_W, 60)
para(tf, "Off-axis columns move by at most 16% against on-axis moves of "
         "45\u2013116%", 14.5, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=6)
para(tf, "Thesis Table 4.12, n = 5,000.  Bold marks the targeted axis; these are "
         "axis values, not scored metrics, so no column carries an arrow.",
     11, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.3)
s.notes_slide.notes_text_frame.text = ("[backup]  Each drive moved its own axis "
    "and left the other three where the baseline put them. Cloc cuts the centroid "
    "by 83 percent, Cscale1500 cuts variance by 65, Crank15 halves effective rank, "
    "Corient doubles subspace overlap, and Clocorient moves both of its targets.")

s = backup(nxt(), "Backup \u2014 location dose\u2013response")
rows = [["Condition", "G\u03bc", "tr \u03a3X", "r", "CLIPScore \u2191",
         "z (step) \u2191", "SPICE \u2191"],
        ["full (unpinned)",           "245.46", "4581.6", "38.61", "0.5433",
         "n/a", "0.0908"],
        ["loc-pinned",                "177.81", "4312.3", "32.32", "0.5797",
         "+14.1", "0.0999"],
        ["loc-drive \u03bb = 0.01",  "80.59",  "4788.0", "27.45", "0.6240",
         "+18.7", "0.1182"],
        ["loc-drive \u03bb = 0.1",   "29.61",  "4791.9", "21.30", "0.6337",
         "+4.3",  "0.1217"]]
booktabs(s, 118, 172, 724, rows, col_w=[1.75, 1.0, 1.05, 0.8, 1.25, 1.1, 1.0],
         size=13.5, head_size=12, row_h=34, head_h=30,
         bold_cells={(1,3),(4,1),(4,2),(3,5),(4,4),(4,6)})
_, tf = _txbox(s, MARGIN_L, 380, CONTENT_W, 80)
para(tf, "Eight-fold closure; gains diminish \u2014 0.0364, 0.0443, then 0.0097",
     15, bold=True, color=TITLE_COL, align=PP_ALIGN.CENTER, first=True,
     space_after=7)
para(tf, "SPICE, which scores against human references rather than a contrastive "
         "model, follows the same progression.", 12.5, color=BODY_COL,
     align=PP_ALIGN.CENTER, space_after=7)
para(tf, "Thesis Table 4.13, n = 5,000.  z is against the preceding row; the "
         "scale pin is active in the lower three conditions, holding variance "
         "within 11%.  G\u03bc, tr \u03a3X and r are axis values, not scored "
         "metrics.", 10.5, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.3)
s.notes_slide.notes_text_frame.text = ("[backup]  How much closure is needed. The "
    "centroid falls eight-fold across four rungs and CLIPScore rises at every "
    "step, but the gains diminish sharply: most of the available gain is realised "
    "by the point where the distance is a third of its original value. Two "
    "caveats: the scale pin only acts on the bottom three rungs, and the "
    "participation ratio co-moves on this ladder, which is why the factorial and "
    "not the ladder carries the shape claim.")

s = backup(nxt(), "Backup — scaling ladder caveat")
_, tf = _txbox(s, MARGIN_L + 40, 232, CONTENT_W - 80, 200)
para(tf, "The ladder has no matched all-pinned run at 1,529 or 2,500 steps.",
     20, bold=True, color=TITLE_COL, align=PP_ALIGN.CENTER, first=True,
     space_after=26)
para(tf, "Only the location drive was carried past 450 steps, so the longer rungs "
         "are compared against the 450-step baseline and against the reference, "
         "not against an equally-trained pinned control.",
     16, color=BODY_COL, align=PP_ALIGN.CENTER, line=1.45)
s.notes_slide.notes_text_frame.text = ("[backup]  An honest caveat. The scaling "
    "ladder has no matched all-pinned run at the longer budgets, so those rungs are "
    "read against the 450-step baseline and the reference, not against an "
    "equally-trained control.")

s = backup(nxt(), "Backup \u2014 the reference's own gap")
rows = [["", "\u0047\u0302\u03bc (normalised)", "O\u2084", "O\u2081\u2086",
         "eff. rank"],
        ["all-pinned baseline", "2.727", "0.0318", "0.0515", "36.99"],
        ["LLaVA-1.5-7B",        "3.250", "0.0026", "0.0086", "3.42"],
        ["chance (q / d)",      "\u2014", "0.0010", "0.0039", "\u2014"]]
booktabs(s, 204, 178, 552, rows, col_w=[1.8, 1.5, 1.0, 1.05, 1.05], size=14,
         head_size=12.5, row_h=36, head_h=32, rule_after=(2,))
_, tf = _txbox(s, MARGIN_L, 344, CONTENT_W, 96)
para(tf, "The reference carries a larger normalised gap than our baseline, yet "
         "captions far better.", 15, bold=True, color=TITLE_COL,
     align=PP_ALIGN.CENTER, first=True, space_after=8)
para(tf, "Training at scale does not close this axis as a by-product \u2014 "
         "deliberately shutting it down creates genuine headroom.",
     13.5, color=MUTED_COL, align=PP_ALIGN.CENTER, space_after=8)
para(tf, "Caveat from the thesis: the reference cloud is extremely anisotropic, "
         "effective rank 3.42 against about 37, which exaggerates its G\u03bc.",
     11, color=MUTED_COL, align=PP_ALIGN.CENTER, line=1.3)
s.notes_slide.notes_text_frame.text = ("[backup]  A fair question is whether scale "
    "alone closes the gap. It does not: the reference carries a larger normalised "
    "centroid gap than our baseline, 3.25 against 2.73, its subspaces sit only two "
    "to three times chance overlap, and it still captions far better. So shutting "
    "the axis down deliberately creates real headroom. The caveat is that the "
    "reference cloud is very anisotropic, which inflates its measured Gmu.")

s = backup(nxt(), "Backup \u2014 the gap in our own model")
place_figure(s, F("gap_panel.png"), 229.0, 117.6, 8,
             W/2, 150, 640, 268, tag="b gap panel")
_, tf = _txbox(s, MARGIN_L, 430, CONTENT_W, 60)
para(tf, "The all-pinned baseline, projected to its first two principal "
         "components", 14.5, color=BODY_COL, align=PP_ALIGN.CENTER, first=True,
     space_after=6)
runs(tf, [("Centroid distance  ", False, MUTED_COL), ("G\u03bc = 177.9", True, ACCENT),
          ("      total variance  ", False, MUTED_COL),
          ("tr \u03a3X = 4254", True, ACCENT)],
     13.5, align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = ("[backup]  The gap as measured in our own "
    "model rather than illustrated: the image cloud and the text centroid, "
    "projected onto the first two principal components of the baseline. The "
    "centroid distance is 177.9 and total variance 4254.")

prs.save(A("presentation.pptx"))
print("\n--- per-slide speaking targets -------------------------------------")
for k in sorted(TIMES):
    print(f"  slide {k:2d}  {TIMES[k]:3d} s" + ("   (divider)" if TIMES[k] == 0 else ""))
talk = sum(TIMES.values())
print(f"\n  main talk, slides 1-{max(TIMES)}: {talk} s = {talk//60}:{talk%60:02d}"
      f"   [window 10:30-11:15]")
print("\n--- smallest rendered type in each figure --------------------------")
for tag, eff in LEGIBILITY:
    print(f"  {tag:24s} {eff:5.1f} pt  {'OK' if eff >= 14 else '<< UNDER 14pt'}")
print(f"\nsaved presentation.pptx  |  {len(prs.slides._sldIdLst)} slides")
