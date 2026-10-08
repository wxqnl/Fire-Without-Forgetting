"""Fig. 2 (main figure): Spike-Driven Dreamer at one environment step (a) and where memory lives (b).

Writes fig/overview.svg and, with rsvg-convert, fig/overview.pdf (vector). Units: 1 = 0.1 mm, so the
canvas is 175 mm wide, the CVPR text width; font sizes of 21-27 units are 6-7.7 pt.
usage: python fig/overview_svg.py [path to rsvg-convert]
"""
import math
import pathlib
import subprocess
import sys

W, H = 1750, 540
SANS = "Nimbus Sans, Helvetica, Arial, sans-serif"
SERIF = "Nimbus Roman, Times New Roman, Times, serif"
INK, GREY, LIGHT = "#222222", "#6B7280", "#9CA3AF"
INTEG, RESON, RESET, OURS = "#0072B2", "#E69F00", "#C0392B", "#D55E00"
FILL = {"enc": "#F3F4F6", "pdn": "#FDF1EA", "lat": "#EEF4FA", "act": "#EEF6F0"}
STROKE = {"enc": "#A3A8B0", "pdn": "#E2A07C", "lat": "#8FB3D9", "act": "#8CC09B"}

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=21, anchor="start", weight="normal", color=INK, style="normal", family=SANS):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{color}" text-anchor="{anchor}">{s}</text>')


def m(base, sub=None, size=23):
    """Inline math: italic serif base and a smaller upright subscript (baseline-shift keeps the baseline)."""
    s = f'<tspan font-family="{SERIF}" font-style="italic" font-size="{size}">{base}</tspan>'
    if sub:
        s += f'<tspan font-family="{SERIF}" font-style="normal" font-size="{size * 0.68:.1f}" baseline-shift="-28%">{sub}</tspan>'
    return s


def rich(x, y, parts, size=21, anchor="start", color=INK, weight="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}" xml:space="preserve">{"".join(parts)}</text>')


def box(x, y, w, h, kind, r=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="2"{d}/>')


def line(x1, y1, x2, y2, color=INK, w=2, arrow=True, dash=None, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = f' marker-end="url(#arr{color[1:]})"' if arrow else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"{d}{a} opacity="{opacity}"/>')


def path(d, color=INK, w=2, arrow=True, fill="none", dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    a = f' marker-end="url(#arr{color[1:]})"' if arrow else ""
    add(f'<path d="{d}" stroke="{color}" stroke-width="{w}" fill="{fill}" stroke-linejoin="round"{ds}{a}/>')


def spikes(x, y, pattern=(1, 1, 1, 1, 1), h=20, dx=6, color=INK, w=2.2):
    """T = 5 frames of a binary spike: a tick per frame (the soma repeats its one spike over the T frames)."""
    for i, on in enumerate(pattern):
        hh = h if on else 3
        add(f'<line x1="{x + i * dx:.1f}" y1="{y:.1f}" x2="{x + i * dx:.1f}" y2="{y - hh:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')


def wave(x1, x2, y, amp=4.5, period=26, color=RESON, w=2.4, arrow=True):
    n = int((x2 - x1) / 2)
    pts = [(x1 + (x2 - x1) * k / n, y + amp * math.sin(2 * math.pi * (x2 - x1) * k / n / period)) for k in range(n + 1)]
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
    path(d, color, w, arrow)


def zigzag(x1, y1, x2, y2, color=RESET, w=2.2, n=7, amp=4):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    pts = [(x1, y1)]
    for k in range(1, n):
        t = k / n
        s = amp if k % 2 else -amp
        pts.append((x1 + dx * t + px * s, y1 + dy * t + py * s))
    pts.append((x2, y2))
    path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), color, w, True)


def soma(x, y, r=13, label=None, fill="#FFFFFF", stroke=INK):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    if label:
        text(x, y + 6, label, 17, "middle", color=GREY)


def dendrite_band(x1, x2, y, gap=11, arrow=True, w=2.4):
    """Two integrator (straight, blue) and two resonant (wavy, orange) channels, top to bottom."""
    for k, kind in enumerate(("i", "i", "r", "r")):
        yy = y + k * gap
        if kind == "i":
            line(x1, yy, x2, yy, INTEG, w, arrow)
        else:
            wave(x1, x2, yy, color=RESON, w=w, arrow=arrow)


# ------------------------------------------------------------------ defs
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W / 10}mm" height="{H / 10}mm" viewBox="0 0 {W} {H}">')
add("<defs>")
for c in (INK, INTEG, RESON, RESET, OURS, GREY):
    add(f'<marker id="arr{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>')
add('<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE3F5"/><stop offset="0.62" stop-color="#E9F2FA"/>'
    '<stop offset="0.62" stop-color="#C9B79C"/><stop offset="1" stop-color="#B7A385"/></linearGradient>')
add("</defs>")
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ================================================================== (a) agent at one step
text(0, 26, "(a) Spike-Driven Dreamer at one environment step", 25, weight="bold")

# observation and spiking encoder (top row)
add('<rect x="10" y="56" width="100" height="100" rx="6" fill="url(#sky)" stroke="#8C8C8C" stroke-width="1.5"/>')
path("M 60 118 L 60 90", INK, 4, False); add('<circle cx="60" cy="82" r="7" fill="#222"/>')
path("M 60 118 L 48 136 L 44 152", INK, 4, False); path("M 60 118 L 72 135 L 82 147", INK, 4, False)
rich(60, 180, [m("x", "t")], anchor="middle")
line(114, 106, 156, 106)
box(160, 72, 180, 68, "enc")
text(250, 100, "spiking encoder", 20, "middle", weight="bold")
text(250, 124, "conv, LIF neurons", 17, "middle", color=GREY)
line(340, 106, 372, 106, INK, 2, False)
spikes(378, 116, (1, 0, 1, 1, 0), h=16, dx=5, w=1.8); rich(408, 114, [m("e", "t")], size=21)
path("M 430 106 L 785 106 L 785 186")

# PDN core: two layers (depth, vertical) unrolled in time (horizontal)
CX0, CX1 = 150, 672
box(CX0, 196, CX1 - CX0, 282, "pdn", r=12)
text(CX0 + 12, 220, "PDN core", 21, weight="bold", color=OURS)
text(CX1 - 12, 220, "dendrites are never reset", 17, "end", color=GREY, style="italic")
neur_x = [268, 328, 388, 448, 508, 568]
for li, (y0, name) in enumerate(((226, "layer 2"), (346, "layer 1"))):
    text(CX0 + 12, y0 + 36, name, 18, color=GREY)
    for x in neur_x:
        line(x, y0 + 43, x, y0 + 58, "#555555", 1.6, False)
        soma(x, y0 + 30)
    dendrite_band(CX0 + 46, CX1 - 40, y0 + 62, gap=11)
    if li == 1:  # time labels once; the layer-2 band's right end feeds the actor-critic
        text(CX0 + 12, y0 + 85, "t−1", 15, color=GREY)
        text(CX1 - 10, y0 + 85, "t+1", 15, "end", color=GREY)
# layer-1 spikes up to the layer-2 dendrites
for x in neur_x:
    line(x + 16, 360, x + 16, 332, INK, 1.6)
# input spikes of (z_{t-1}, a_{t-1}) into layer 1
for x in neur_x:
    line(x + 16, 500, x + 16, 455, INK, 1.6)
path(f"M {neur_x[0] + 16} 500 L {neur_x[-1] + 16} 500", INK, 1.6, False)
path("M 418 500 L 418 510", INK, 1.6, False)
rich(418, 532, ["spikes of ", m("z", "t−1", 21), " and ", m("a", "t−1", 21)], anchor="middle", size=18)

# h_t out of layer 2 to the latent
line(584, 256, 706, 256)
spikes(612, 249, h=15, dx=5, w=1.8); rich(644, 247, [m("h", "t", 21)])

# posterior / prior
box(706, 186, 162, 140, "lat")
text(787, 210, "latent", 20, "middle", weight="bold")
text(787, 236, "posterior", 16, "middle", color=GREY)
rich(787, 258, [m("q", None, 20), "(", m("z", "t", 20), " | ", m("h", "t", 20), ", ", m("e", "t", 20), ")"], anchor="middle", size=19)
text(787, 286, "prior", 16, "middle", color=GREY)
rich(787, 308, [m("p", None, 20), "(", m("z", "t", 20), " | ", m("h", "t", 20), ")"], anchor="middle", size=19)
line(868, 256, 930, 256, INK, 2, False)
spikes(878, 249, (0, 1, 0, 0, 1), h=15, dx=5, w=1.8); rich(890, 280, [m("z", "t", 20)], anchor="middle")

# heads and actor-critic, both read (z_t, h_t)
path("M 930 256 L 930 132 L 950 132")
path("M 930 256 L 930 350 L 950 350")
box(950, 60, 158, 134, "enc")
text(1029, 84, "heads", 20, "middle", weight="bold")
text(1029, 112, "image decoder", 17, "middle", color=INK)
text(1029, 138, "reward", 17, "middle", color=INK)
text(1029, 164, "continuation", 17, "middle", color=INK)
box(950, 292, 158, 118, "act")
text(1029, 316, "actor-critic", 20, "middle", weight="bold")
rich(1029, 344, [m("π", None, 20), "(", m("a", "t", 20), " | ·),  ", m("v", None, 20), "(·)"], anchor="middle", size=19)
text(1029, 370, "reads z, h and the", 15, "middle", color=GREY)
text(1029, 388, "dendritic spikes", 15, "middle", color=GREY)
line(1029, 410, 1029, 446); rich(1040, 444, [m("a", "t", 20)])
# dendritic ON/OFF spikes of layer 2 to the actor-critic
path("M 636 304 C 700 304, 700 392, 752 392 L 950 392", OURS, 2)
spikes(800, 386, (1, 1, 1, 0, 0), h=14, dx=5, w=1.8, color=OURS)
spikes(832, 386, (1, 0, 0, 0, 0), h=14, dx=5, w=1.8, color=OURS)
text(848, 414, "ON/OFF dendritic spikes", 15, "middle", color=OURS)

# legend (bottom right of panel a)
ly = 526
spikes(706, ly + 2, h=13, dx=4.5, w=1.6); text(734, ly, "spike frames (T = 5)", 15, color=GREY)
line(884, ly - 5, 914, ly - 5, INTEG, 2.4, False); text(920, ly, "integrator", 15, color=GREY)
wave(996, 1026, ly - 5, color=RESON, arrow=False); text(1032, ly, "resonant", 15, color=GREY)

# ================================================================== (b) where memory lives
X0 = 1168
add(f'<line x1="{X0 - 24}" y1="40" x2="{X0 - 24}" y2="530" stroke="#E5E7EB" stroke-width="2"/>')
text(X0, 26, "(b) Where memory lives", 25, weight="bold")
FW, FH, GAP = 226, 176, 100
FX = (X0, X0 + FW + GAP)


def frame(x, y, label):
    add(f'<rect x="{x}" y="{y}" width="{FW}" height="{FH}" rx="8" fill="none" stroke="#C4C8CE" stroke-width="1.6" stroke-dasharray="7 5"/>')
    text(x + 8, y + 20, label, 15, color=GREY)


# multi-compartment neuron (Spiking-WM): cleared at each step, dendrites reset by a spike
text(X0, 60, "Spiking-WM: multi-compartment neuron", 19, weight="bold")
y = 72
for x, lab in zip(FX, ("step t−1", "step t")):
    frame(x, y, lab)
    cx = x + FW / 2
    soma(cx, y + 64, 18, fill="#F5F5F5")
    spikes(cx - 12, y + 38, h=15, dx=6)
    for side in (-1, 1):
        ex, ey = cx + side * 58, y + 140
        add(f'<ellipse cx="{ex}" cy="{ey}" rx="38" ry="16" fill="#F5F5F5" stroke="{INK}" stroke-width="2"/>')
        text(ex, ey + 5, "basal" if side < 0 else "apical", 14, "middle", color=GREY)
        line(cx + side * 10, y + 80, ex - side * 6, ey - 15, INK, 2, False)
        zigzag(cx + side * 20, y + 74, ex + side * 16, ey - 17)
    text(cx, y + 170, "reset on spike", 13, "middle", color=RESET)
# no state crosses the step boundary
gx = X0 + FW + GAP / 2
line(X0 + FW + 6, y + 140, X0 + FW + GAP - 8, y + 140, LIGHT, 2, True, dash="5 4")
text(gx, y + 147, "✕", 22, "middle", weight="bold", color=RESET)
text(gx, y + 170, "cleared", 14, "middle", color=RESET)
# only the spikes of step t-1 reach step t
path(f"M {FX[0] + FW / 2 + 22} {y + 30} C {gx - 20} {y + 30}, {gx + 10} {y + 40}, {FX[1] + FW / 2 - 66} {y + 122}", INK, 2)
text(gx, y + 82, "only spikes", 14, "middle", color=INK)
text(gx, y + 98, "cross", 14, "middle", color=INK)

# persistent dendritic neuron (ours): channels carried across steps, stateless soma
text(X0, 296, "SD-Dreamer: persistent dendritic neuron", 19, weight="bold", color=OURS)
y = 308
for x, lab in zip(FX, ("step t−1", "step t")):
    frame(x, y, lab)
    cx = x + FW / 2
    soma(cx, y + 64, 18)
    spikes(cx - 12, y + 38, h=15, dx=6)
    text(cx + 26, y + 70, "stateless", 14, color=GREY)
    for side in (-1, 1):
        line(cx + side * 26, y + 120, cx + side * 10, y + 84, "#555555", 1.8)
    rich(cx - 34, y + 110, ["∑", m("C", "m", 17), m("d", "m", 17)], anchor="end", size=15, color=GREY)
dendrite_band(X0 - 8, X0 + 2 * FW + GAP + 6, 432, gap=13, w=2.6)
text(gx, 422, "never reset", 14, "middle", color=INK)
text(X0, 524, "integrator: what", 15, color=INTEG)
text(X0 + 128, 524, "resonant: when (phase)", 15, color=RESON)

add("</svg>")
here = pathlib.Path(__file__).parent
svg = here / "overview.svg"
svg.write_text("\n".join(out))
rsvg = sys.argv[1] if len(sys.argv) > 1 else "rsvg-convert"
subprocess.run([rsvg, "-f", "pdf", "-o", str(here / "overview.pdf"), str(svg)], check=True)
subprocess.run([rsvg, "-f", "png", "-d", "300", "-p", "300", "-o", str(here / "overview_preview.png"), str(svg)], check=True)
print("wrote", svg, here / "overview.pdf")
