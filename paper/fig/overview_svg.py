"""Fig. 2 (main figure): Spike-Driven Dreamer at one environment step (a) and where memory lives (b).

Final model (EXPERIMENTS.md §19.27): two PDN layers, one integrator and one resonant channel per neuron, a stateless
soma that sends a spike count over the T = 5 simulation steps (thermometer code), spike feedback into the dendrites,
and the actor-critic reading the second-layer dendrites as ON/OFF spikes.
Writes fig/overview.svg and, with rsvg-convert, fig/overview.pdf (vector). Units: 1 = 0.1 mm, so the canvas is
175 mm wide, the CVPR text width; text is 19-26 units (5.4-7.4 pt).
usage: python fig/overview_svg.py [path to rsvg-convert]
"""
import math
import pathlib
import subprocess
import sys

W, H = 1750, 560
SANS = "Nimbus Sans, Helvetica, Arial, sans-serif"
SERIF = "Nimbus Roman, Times New Roman, Times, serif"
INK, GREY, LIGHT = "#1F2328", "#5F6670", "#A7ADB5"
INTEG, RESON, RESET, OURS, FB = "#0072B2", "#E69F00", "#C0392B", "#D55E00", "#7A4FB5"
FILL = {"enc": "#F4F5F7", "pdn": "#FDF3EC", "lat": "#EDF3FA", "act": "#EDF6EF", "frame": "#FFFFFF"}
STROKE = {"enc": "#B3B8C0", "pdn": "#E5A884", "lat": "#94B7DA", "act": "#94C6A1", "frame": "#C9CDD3"}
S_TXT, S_LAB, S_HEAD, S_TITLE = 19, 21, 23, 26

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=S_LAB, anchor="start", weight="normal", color=INK, style="normal", family=SANS):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{color}" text-anchor="{anchor}">{s}</text>')


def m(base, sub=None, size=24):
    """Inline math: italic serif base and a smaller upright subscript (baseline-shift keeps the baseline)."""
    s = f'<tspan font-family="{SERIF}" font-style="italic" font-size="{size}">{base}</tspan>'
    if sub:
        s += f'<tspan font-family="{SERIF}" font-style="normal" font-size="{size * 0.7:.1f}" baseline-shift="-28%">{sub}</tspan>'
    return s


def rich(x, y, parts, size=S_LAB, anchor="start", color=INK, weight="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}" xml:space="preserve">{"".join(parts)}</text>')


def box(x, y, w, h, kind, r=12, dash=None, sw=2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, color=INK, w=2, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = f' marker-end="url(#arr{color[1:]})"' if arrow else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"{d}{a}/>')


def path(d, color=INK, w=2, arrow=True, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    a = f' marker-end="url(#arr{color[1:]})"' if arrow else ""
    add(f'<path d="{d}" stroke="{color}" stroke-width="{w}" fill="none" stroke-linejoin="round" stroke-linecap="round"{ds}{a}/>')


def spikes(x, y, n=5, T=5, h=20, dx=6.5, color=INK, w=2.4):
    """Thermometer code over the T simulation steps: the first n frames carry a spike, the others are silent."""
    for i in range(T):
        hh = h if i < n else 3
        add(f'<line x1="{x + i * dx:.1f}" y1="{y:.1f}" x2="{x + i * dx:.1f}" y2="{y - hh:.1f}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"/>')


def binary(x, y, pattern, h=16, dx=6, color=INK, w=2.2):
    for i, on in enumerate(pattern):
        hh = h if on else 3
        add(f'<line x1="{x + i * dx:.1f}" y1="{y:.1f}" x2="{x + i * dx:.1f}" y2="{y - hh:.1f}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"/>')


def wave(x1, x2, y, amp=5, period=28, color=RESON, w=2.6, arrow=True):
    n = int((x2 - x1) / 2)
    pts = [(x1 + (x2 - x1) * k / n, y + amp * math.sin(2 * math.pi * (x2 - x1) * k / n / period)) for k in range(n + 1)]
    path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts), color, w, arrow)


def channels(x1, x2, y, gap=16, w=2.8, arrow=True):
    """The two dendritic channels of a PDN neuron: integrator (straight, blue) and resonant (wavy, orange)."""
    line(x1, y, x2, y, INTEG, w, arrow)
    wave(x1, x2, y + gap, color=RESON, w=w, arrow=arrow)


def zigzag(x1, y1, x2, y2, color=RESET, w=2.4, n=7, amp=4.5):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    px, py = -dy / L, dx / L
    pts = [(x1, y1)] + [(x1 + dx * k / n + px * (amp if k % 2 else -amp), y1 + dy * k / n + py * (amp if k % 2 else -amp))
                        for k in range(1, n)] + [(x2, y2)]
    path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), color, w, True)


def soma(x, y, r=15, fill="#FFFFFF", stroke=INK, w=2.2):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')


# ------------------------------------------------------------------ defs
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W / 10}mm" height="{H / 10}mm" viewBox="0 0 {W} {H}">')
add("<defs>")
for c in (INK, INTEG, RESON, RESET, OURS, GREY, FB):
    add(f'<marker id="arr{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>')
add('<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CFE3F5"/><stop offset="0.62" stop-color="#E9F2FA"/>'
    '<stop offset="0.62" stop-color="#C9B79C"/><stop offset="1" stop-color="#B7A385"/></linearGradient>')
add("</defs>")
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ================================================================== (a) agent at one step
text(0, 28, "(a) SD-Dreamer at one environment step", S_TITLE, weight="bold")

# observation and spiking encoder (top row)
add('<rect x="8" y="60" width="96" height="96" rx="7" fill="url(#sky)" stroke="#8C8C8C" stroke-width="1.5"/>')
path("M 56 120 L 56 92", INK, 4, False); add('<circle cx="56" cy="84" r="7" fill="#222"/>')
path("M 56 120 L 45 137 L 41 151", INK, 4, False); path("M 56 120 L 67 136 L 77 148", INK, 4, False)
rich(56, 186, [m("x", "t")], anchor="middle")
line(108, 108, 150, 108)
box(154, 74, 196, 70, "enc")
text(252, 103, "spiking encoder", S_HEAD, "middle", weight="bold")
text(252, 129, "LIF neurons", S_TXT, "middle", color=GREY)
line(350, 108, 384, 108, INK, 2, False)
binary(390, 118, (1, 0, 1, 1, 0), h=16, dx=5.5, w=2); rich(420, 116, [m("e", "t", 23)])
path("M 444 108 L 806 108 L 806 206")

# PDN core: two layers (depth, vertical), dendrites drawn along time (horizontal)
CX0, CX1, CY0, CY1 = 140, 706, 206, 466
box(CX0, CY0, CX1 - CX0, CY1 - CY0, "pdn", r=14)
text(CX0 + 14, CY0 + 30, "PDN core", S_HEAD, weight="bold", color=OURS)
text(CX1 - 14, CY0 + 30, "PDM: dendrites are never reset", S_TXT, "end", color=GREY)
NX = [292, 372, 452, 532]
BX0, BX1 = CX0 + 52, CX1 - 74
LAYERS = ((262, "layer 2"), (372, "layer 1"))
for li, (ys, name) in enumerate(LAYERS):
    text(CX0 + 14, ys + 6, name, S_TXT, color=GREY)
    for x in NX:
        line(x, ys + 15, x, ys + 40, "#555555", 1.8, False)
        soma(x, ys)
        spikes(x + 20, ys - 4, n=(3, 5, 2, 4)[NX.index(x)] if li == 0 else (4, 2, 5, 3)[NX.index(x)], h=15, dx=4.2, w=1.8)
    channels(BX0, BX1, ys + 41)
    # spike feedback: the layer's spikes enter its own dendrites at the next step
    fx = NX[-1]
    path(f"M {fx + 15} {ys + 6} C {fx + 44} {ys + 8}, {fx + 52} {ys + 20}, {fx + 52} {ys + 34}", FB, 2.2)
    text(fx + 62, ys + 30, "R", S_TXT + 1, color=FB, style="italic", family=SERIF)
text(BX0 - 8, LAYERS[1][0] + 47, "t−1", S_TXT - 2, "end", color=GREY)
text(BX1 + 10, LAYERS[1][0] + 47, "t", S_TXT - 2, color=GREY)
# layer-1 spikes up to the layer-2 dendrites
for x in NX:
    line(x + 28, LAYERS[1][0] - 22, x + 28, LAYERS[0][0] + 66, INK, 1.8)
# input spikes of (z_{t-1}, a_{t-1}) into layer 1
yb = CY1 + 12
for x in NX:
    line(x + 28, yb, x + 28, LAYERS[1][0] + 66, INK, 1.8)
path(f"M {NX[0] + 28} {yb} L {NX[-1] + 28} {yb}", INK, 1.8, False)
# every write into the channels uses the normalized write gain
text(NX[0] + 18, yb - 22, "NWG write", S_TXT - 1, "end", color=GREY)
path(f"M {(NX[0] + NX[-1]) / 2 + 28} {yb} L {(NX[0] + NX[-1]) / 2 + 28} {yb + 10}", INK, 1.8, False)
rich((NX[0] + NX[-1]) / 2 + 28, yb + 34, ["spikes of ", m("z", "t−1", 22), " and ", m("a", "t−1", 22)], anchor="middle", size=S_TXT)

# h_t out of layer 2 to the latent
yh = LAYERS[0][0] - 10
line(NX[-1] + 46, yh, 732, yh)
rich(721, yh + 28, [m("h", "t", 23)], anchor="middle")

# posterior / prior
box(736, 206, 160, 146, "lat")
text(816, 234, "latent", S_HEAD, "middle", weight="bold")
text(816, 262, "posterior", S_TXT, "middle", color=GREY)
rich(816, 286, [m("q", None, 22), "(", m("z", "t", 22), " | ", m("h", "t", 22), ", ", m("e", "t", 22), ")"], anchor="middle", size=S_LAB)
text(816, 314, "prior", S_TXT, "middle", color=GREY)
rich(816, 338, [m("p", None, 22), "(", m("z", "t", 22), " | ", m("h", "t", 22), ")"], anchor="middle", size=S_LAB)
line(896, 279, 950, 279, INK, 2, False)
binary(906, 271, (0, 1, 0, 0, 1), h=15, dx=5, w=1.9); rich(922, 304, [m("z", "t", 22)], anchor="middle")

# heads and actor-critic read (z_t, h_t)
path("M 950 279 L 950 132 L 976 132")
path("M 950 279 L 950 376 L 976 376")
box(980, 62, 156, 142, "enc")
text(1058, 92, "heads", S_HEAD, "middle", weight="bold")
for k, s in enumerate(("decoder", "reward", "continuation")):
    text(1058, 122 + 26 * k, s, S_TXT, "middle")
box(980, 316, 156, 140, "act")
text(1058, 346, "actor-critic", S_HEAD, "middle", weight="bold")
rich(1058, 376, [m("π", None, 22), "(", m("a", "t", 22), " | ·),  ", m("v", None, 22), "(·)"], anchor="middle", size=S_LAB)
text(1058, 404, "reads z, h and the", S_TXT - 1, "middle", color=GREY)
text(1058, 426, "dendritic spikes", S_TXT - 1, "middle", color=GREY)
line(1058, 456, 1058, 494); rich(1070, 492, [m("a", "t", 22)])
# dendritic ON/OFF spikes of layer 2 to the actor-critic
yd = LAYERS[0][0] + 49
path(f"M {BX1 + 4} {yd} C {BX1 + 70} {yd}, {BX1 + 60} 432, {BX1 + 130} 432 L 976 432", OURS, 2.4)
binary(800, 425, (1, 1, 1, 0, 0), h=14, dx=5, w=1.9, color=OURS)
binary(834, 425, (1, 0, 0, 0, 0), h=14, dx=5, w=1.9, color=OURS)
text(868, 462, "ON/OFF dendritic spikes", S_TXT - 1, "middle", color=OURS)

# legend (bottom of panel a, right of the input label)
ly = 548
spikes(624, ly + 2, n=3, h=14, dx=4.4, w=1.8); text(652, ly, "spike count", S_TXT - 1, color=GREY)
line(770, ly - 6, 800, ly - 6, INTEG, 2.6, False); text(806, ly, "integrator", S_TXT - 1, color=GREY)
wave(902, 932, ly - 6, color=RESON, arrow=False); text(938, ly, "resonant", S_TXT - 1, color=GREY)
line(1028, ly - 6, 1058, ly - 6, FB, 2.2); text(1064, ly, "feedback", S_TXT - 1, color=GREY)

# ================================================================== (b) where memory lives
X0 = 1180
add(f'<line x1="{X0 - 22}" y1="44" x2="{X0 - 22}" y2="548" stroke="#E5E7EB" stroke-width="2"/>')
text(X0, 28, "(b) Where memory lives", S_TITLE, weight="bold")
FW, FH, GAP = 218, 178, 116
FX = (X0, X0 + FW + GAP)
gx = X0 + FW + GAP / 2


def frame(x, y, label):
    box(x, y, FW, FH, "frame", r=10, dash="7 5", sw=1.6)
    text(x + 10, y + 24, label, S_TXT - 1, color=GREY)


# multicompartment neuron (Spiking-WM): cleared at each step, dendrites reset by a spike
text(X0, 66, "Spiking-WM: multicompartment neuron", S_LAB, weight="bold")
y = 80
for x, lab in zip(FX, ("step t−1", "step t")):
    frame(x, y, lab)
    cx = x + FW / 2 + 8
    soma(cx, y + 66, 18, fill="#F3F4F6")
    binary(cx - 12, y + 40, (1, 1, 1, 1, 1), h=15, dx=6)
    for side in (-1, 1):
        ex, ey = cx + side * 56, y + 142
        add(f'<ellipse cx="{ex}" cy="{ey}" rx="40" ry="17" fill="#F3F4F6" stroke="{INK}" stroke-width="2"/>')
        text(ex, ey + 6, "basal" if side < 0 else "apical", S_TXT - 2, "middle", color=GREY)
        line(cx + side * 10, y + 83, ex - side * 6, ey - 16, INK, 2, False)
        zigzag(cx + side * 21, y + 76, ex + side * 18, ey - 18)
    text(cx, y + FH - 6 + 26, "reset on spike", S_TXT - 2, "middle", color=RESET)
line(X0 + FW + 6, y + 142, X0 + FW + GAP - 8, y + 142, LIGHT, 2, True, dash="5 4")
text(gx, y + 150, "✕", 24, "middle", weight="bold", color=RESET)
text(gx, y + 176, "cleared", S_TXT - 2, "middle", color=RESET)
path(f"M {FX[0] + FW / 2 + 32} {y + 32} C {gx - 20} {y + 32}, {gx + 10} {y + 44}, {FX[1] + FW / 2 - 60} {y + 124}", INK, 2)
text(gx, y + 100, "only spikes", S_TXT - 2, "middle")
text(gx, y + 120, "cross", S_TXT - 2, "middle")

# persistent dendritic neuron (ours): channels carried across steps, stateless soma, spike count, feedback
text(X0, 324, "SD-Dreamer: persistent dendritic neuron", S_LAB, weight="bold", color=OURS)
y = 338
for x, lab, n in zip(FX, ("step t−1", "step t"), (4, 2)):
    frame(x, y, lab)
    cx = x + FW / 2 + 8
    soma(cx, y + 62, 18)
    spikes(cx - 12, y + 36, n=n, h=15, dx=6)
    text(cx + 26, y + 68, "stateless", S_TXT - 2, color=GREY)
    for side in (-1, 1):
        line(cx + side * 28, y + 122, cx + side * 10, y + 80, "#555555", 2)
    rich(cx + 36, y + 112, ["∑", m("C", None, 19), m("d", None, 19)], size=S_TXT - 2, color=GREY)
channels(X0 - 10, X0 + 2 * FW + GAP + 10, y + 128, gap=18, w=2.8)
text(gx, y + 100, "PDM:", S_TXT - 2, "middle")
text(gx, y + 120, "never reset", S_TXT - 2, "middle")
# the spikes of step t-1 feed back into the dendrites of step t; they add to the channels, they do not clear them
path(f"M {FX[0] + FW / 2 + 30} {y + 26} C {gx + 30} {y + 26}, {FX[1] + 36} {y + 36}, {FX[1] + 40} {y + 118}", FB, 2.2)
text(gx, y + 16, "feedback", S_TXT - 2, "middle", color=FB)
text(X0, H - 12, "integrator: what", S_TXT - 1, color=INTEG)
text(X0 + 150, H - 12, "resonant: also when (phase)", S_TXT - 1, color=RESON)

add("</svg>")
here = pathlib.Path(__file__).parent
svg = here / "overview.svg"
svg.write_text("\n".join(out))
rsvg = sys.argv[1] if len(sys.argv) > 1 else "rsvg-convert"
subprocess.run([rsvg, "-f", "pdf", "-o", str(here / "overview.pdf"), str(svg)], check=True)
subprocess.run([rsvg, "-f", "png", "-d", "300", "-p", "300", "-o", str(here / "overview_preview.png"), str(svg)], check=True)
print("wrote", svg, here / "overview.pdf")
