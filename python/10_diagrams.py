"""Explainer diagrams for the story page, drawn with the family's famdiagram
(../basketball-analysis-tools/famdiagram.py) and saved to figures/<ref>.svg
(plus a <ref>_narrow.svg phone layout). report_builder.py inlines them.

Every number drawn is read from a committed output; nothing is typed here.

Diagrams:
  dg1_two_skills     how a shot gets a price, and how a player's points per
                     shot split into selection and making, on the season's
                     top shot-maker and top shot-selector among qualifying
                     players (output/shot_prices.csv, output/player_table.csv)
  dg2_trust_diet     how the persistence finding becomes the projection rule:
                     what repeats, then the shrinkage on a high-volume and a
                     low-volume shooter, then the out-of-sample result
                     (output/persistence.csv, output/projection.csv,
                     output/projection_hyperparams.csv,
                     output/projection_metrics.csv)
  dg3_clock_split    end-of-period shots against normal play, split into a
                     selection step and an execution step
                     (output/clock_decomposition.csv)

Run:     python python/10_diagrams.py
--check: rebuild every SVG in memory and fail if a committed one differs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))
import famdiagram as fd  # noqa: E402

MIN_FGA = 300                      # the qualifying line 04_findings.py uses
PAIR = "2023-24 to 2024-25"        # the season pair the story quotes first
BASE = "37s+"                      # normal play in clock_decomposition.csv
RUSH = "00-04s (rush)"
TFO = "25-36s (2-for-1 window)"
SHOT_LABEL = {"dunk": "Dunk", "long two (mid-range, 16+ ft)": "Long two, 16+ ft",
              "jumper": "Jumper", "layup": "Layup", "alley_oop": "Alley-oop",
              "floater": "Floater", "hook": "Hook", "off_dribble": "Off the dribble",
              "post_jumper": "Post jumper", "putback": "Putback"}


def wrap(s: str, width: float, size: float) -> list[str]:
    """Greedy word wrap on an estimated glyph width."""
    per = size * 0.55
    out, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if cur and len(t) * per > width:
            out.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        out.append(cur)
    return out


def lines(d: fd.Diagram, x, y, s, width, size, gap=None, **kw) -> float:
    """Draw wrapped text; return the baseline after the last line."""
    gap = gap or size * 1.3
    for i, ln in enumerate(wrap(s, width, size)):
        d.text(x, y + i * gap, ln, size=size, **kw)
    return y + i * gap


def per100(v: float) -> str:
    return f"{v * 100:+.1f}"


def short(name: str) -> str:
    return name.split(" ", 1)[1] if " " in name else name


# ---------------------------------------------------------------------------
# 1. From a shot's price to two skills
# ---------------------------------------------------------------------------

def dg_two_skills(layout: str) -> fd.Diagram:
    prices = pl.read_csv(OUT / "shot_prices.csv").sort("xpts", descending=True)
    hi, lo = prices.row(0, named=True), prices.row(-1, named=True)
    allp = pl.read_csv(OUT / "player_table.csv", null_values=["NA"])
    league = allp["pts"].sum() / allp["fga"].sum()
    q = allp.filter(pl.col("fga") >= MIN_FGA)
    maker = q.sort("making_100", descending=True).row(0, named=True)
    picker = q.sort("xpps", descending=True).row(0, named=True)
    title = (f"How a shot is priced, and how {short(maker['PLAYER_NAME'])} and "
             f"{short(picker['PLAYER_NAME'])} split into selection and making")
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    # ---- step 1: the price of one shot
    d = fd.Diagram(W, 0, title)
    d.text(pad, 26, "1. Every shot gets a price", size=s + 2, weight="600")
    boxes = [("The shot", "where it was taken, distance, angle, and action family"),
             ("The model", "the chance a shot like it goes in (xMake)"),
             ("The price", "xMake times 2 or 3: expected points")]
    if wide:
        bw, bh, gap, y0 = 220, 94, 30, 44
        pos = [(pad + i * (bw + gap), y0) for i in range(3)]
    else:
        bw, bh, gap, y0 = W - 2 * pad, 62, 22, 42
        pos = [(pad, y0 + i * (bh + gap)) for i in range(3)]
    for i, ((head, body), (bx, by)) in enumerate(zip(boxes, pos)):
        d.rect(bx, by, bw, bh, fill="surface2", stroke="rule", rx=8, sw=1)
        d.text(bx + 12, by + s + 8, head, size=s, weight="600")
        lines(d, bx + 12, by + 2 * s + 14, body, bw - 24, s, fill="ink2")
        if i < 2:
            if wide:
                d.line(bx + bw + 4, by + bh / 2, bx + bw + gap - 4, by + bh / 2,
                       stroke="ink3", sw=1.6, arrow=True)
            else:
                d.line(bx + bw / 2, by + bh + 3, bx + bw / 2, by + bh + gap - 3,
                       stroke="ink3", sw=1.6, arrow=True)
    py = pos[-1][1] + bh + (30 if wide else 28)
    tail = (f"Dearest shot type: {SHOT_LABEL.get(hi['shot'], hi['shot']).lower()}, "
            f"{hi['xpts']:.2f} expected points. Cheapest: "
            f"{SHOT_LABEL.get(lo['shot'], lo['shot']).lower()}, {lo['xpts']:.2f}.")
    py = lines(d, pad, py, tail, W - 2 * pad, s, fill="ink2")
    # ---- step 2: one player's points per shot, split in two
    y2 = py + (44 if wide else 40)
    d.text(pad, y2, "2. Add up a player's shots", size=s + 2, weight="600")
    sub = ("Points per shot, against the league average. Selection: the price of "
           "the shots he takes. Making: what he scores above that price.")
    y2 = lines(d, pad, y2 + s + 10, sub, W - 2 * pad, s, fill="ink3")
    vals = [league] + [r[k] for r in (maker, picker) for k in ("xpps", "pps")]
    lo_v = min(vals) - 0.06
    hi_v = max(vals) + 0.06
    ax0 = pad + (150 if wide else 0)
    ax1 = W - pad
    X = lambda v: ax0 + (v - lo_v) / (hi_v - lo_v) * (ax1 - ax0)  # noqa: E731
    rows = [maker, picker]
    rh = 74 if wide else 92
    top = y2 + (44 if wide else 40)
    for i, r in enumerate(rows):
        ry = top + i * rh
        sel = r["xpps"] - league
        mk = r["pps"] - r["xpps"]
        if wide:
            d.text(pad, ry + 20, r["PLAYER_NAME"], size=s, weight="600")
            d.text(pad, ry + 20 + s + 4, f"{r['fga']:,} shots", size=s, fill="ink3")
            by = ry + 8
        else:
            d.text(pad, ry + 12, f"{r['PLAYER_NAME']}, {r['fga']:,} shots", size=s,
                   weight="600")
            by = ry + 24
        # selection: league average to his price
        x_a, x_b = X(league), X(r["xpps"])
        d.rect(min(x_a, x_b), by, abs(x_b - x_a), 14, fill="nba", rx=2)
        # making: his price to what he scored
        x_c = X(r["pps"])
        d.rect(min(x_b, x_c), by + 18, abs(x_c - x_b), 14,
               fill="c3" if mk >= 0 else "c8", rx=2)
        d.circle(x_c, by + 25, 4, fill="ink")
        d.line(X(league), by - 5, X(league), by + 37, stroke="ink2", sw=1.4, dash="4 3")
        sel_t = f"selection {per100(sel)}"
        mk_t = f"making {per100(mk)}"
        if wide:
            right = sel >= 0 and max(x_a, x_b) + 6 + len(sel_t) * s * 0.58 < W
            d.text(max(x_a, x_b) + 6 if right else min(x_a, x_b) - 6, by + 12, sel_t,
                   size=s, fill="nba", anchor="start" if right else "end",
                   weight="600")
            d.text(max(x_b, x_c) + 10 if mk >= 0 else min(x_b, x_c) - 6, by + 30, mk_t,
                   size=s, fill="c3" if mk >= 0 else "c8",
                   anchor="start" if mk >= 0 else "end", weight="600")
        else:
            d.text(pad, by + 50, f"{sel_t},", size=s, fill="nba", weight="600")
            d.text(pad + len(sel_t) * s * 0.58 + 8, by + 50, mk_t, size=s,
                   fill="c3" if mk >= 0 else "c8", weight="600")
    yb = top + len(rows) * rh + (2 if wide else -14)
    d.text(X(league) + (0 if wide else -4), top - 12, f"league average {league:.3f}",
           size=s, fill="ink2", anchor="middle" if wide else "start")
    d.line(ax0, yb, ax1, yb, stroke="rule", sw=1)
    step = 0.1
    t = round(lo_v + 0.05, 1)
    while t <= hi_v:
        d.line(X(t), yb, X(t), yb + 4, stroke="rule", sw=1)
        d.text(X(t), yb + s + 6, f"{t:.1f}", size=s, fill="ink3", anchor="middle")
        t = round(t + step, 1)
    d.text(ax1, yb + 2 * s + 12, "points per shot", size=s, fill="ink3", anchor="end")
    foot = "Both steps in points per 100 shots. The dot is what he scored."
    end = lines(d, pad, yb + 2 * s + 44, foot, W - 2 * pad, s, fill="ink3")
    d.h = end + 14
    return d


# ---------------------------------------------------------------------------
# 2. Trust the diet, regress the making
# ---------------------------------------------------------------------------

def dg_trust_diet(layout: str) -> fd.Diagram:
    per = pl.read_csv(OUT / "persistence.csv").filter(pl.col("pair") == PAIR)
    r_sel = per.filter(pl.col("metric") == "xpps")["r"][0]
    r_mk = per.filter(pl.col("metric") == "making_100")["r"][0]
    proj = pl.read_csv(OUT / "projection.csv").filter(pl.col("pair") == PAIR)
    hp = pl.read_csv(OUT / "projection_hyperparams.csv").filter(
        pl.col("season") == PAIR.split(" ")[0]).row(0, named=True)
    met = pl.read_csv(OUT / "projection_metrics.csv").filter(pl.col("pair") == PAIR)
    mae_eb = met.filter(pl.col("method").str.starts_with("eb"))["mae"][0]
    mae_nv = met.filter(pl.col("method").str.starts_with("naive"))["mae"][0]
    gain = 1 - mae_eb / mae_nv
    # The season's top shot-maker (dg1's player) against the most-shrunk shooter.
    hiw = proj.sort("making_100", descending=True).row(0, named=True)
    low = proj.sort("weight").row(0, named=True)
    mu = hp["mu_making_100"]
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    title = (f"Trust the diet, regress the making: {short(hiw['PLAYER_NAME'])} keeps "
             f"{hiw['weight']:.0%} of his making, {short(low['PLAYER_NAME'])} "
             f"{low['weight']:.0%}")
    d = fd.Diagram(W, 0, title)
    d.text(pad, 26, f"1. What repeats from {PAIR}", size=s + 2,
           weight="600")
    cards = [("Selection", r_sel, "nba", "Repeats almost fully: carry it forward whole."),
             ("Making", r_mk, "c3", "Half repeats: shrink it toward the league mean.")]
    if wide:
        cw, ch = (W - 2 * pad - 20) / 2, 86
        cpos = [(pad, 42), (pad + cw + 20, 42)]
    else:
        cw, ch = W - 2 * pad, 80
        cpos = [(pad, 42), (pad, 42 + ch + 14)]
    for (name, r, col, body), (cx, cy) in zip(cards, cpos):
        d.rect(cx, cy, cw, ch, fill="surface2", stroke="rule", rx=8, sw=1)
        d.rect(cx, cy, 6, ch, fill=col, rx=2)
        d.text(cx + 18, cy + s + 10, f"{name}: r = {r:.2f}".replace("0.", "."),
               size=s + 1, weight="600", fill=col)
        lines(d, cx + 18, cy + 2 * s + 20, body, cw - 30, s, fill="ink2")
    y = cpos[-1][1] + ch + (46 if wide else 40)
    d.text(pad, y, "2. Shrink each player's making by how noisy it is", size=s + 2,
           weight="600")
    expl = (f"Share kept = τ² / (τ² + SE²), where τ = {hp['tau']:.1f} points per 100 "
            f"is the league's spread of true making and SE is the player's own noise. "
            f"More shots, smaller SE, more of his number kept.")
    y = lines(d, pad, y + s + 10, expl, W - 2 * pad, s, fill="ink3")
    # making axis
    rows = [hiw, low]
    obs = [r["making_100"] for r in rows]
    ses = [r["making_100_se"] for r in rows]
    lo_v = min(min(o - e for o, e in zip(obs, ses)), mu) - 3
    hi_v = max(max(o + e for o, e in zip(obs, ses)), mu) + 3
    ax0 = pad + (170 if wide else 0)
    ax1 = W - pad - (10 if wide else 4)
    X = lambda v: ax0 + (v - lo_v) / (hi_v - lo_v) * (ax1 - ax0)  # noqa: E731
    top = y + 40
    rh = 70 if wide else 104
    for i, r in enumerate(rows):
        ry = top + i * rh
        if wide:
            d.text(pad, ry + 14, r["PLAYER_NAME"], size=s, weight="600")
            d.text(pad, ry + 14 + s + 4, f"{r['fga']:,} shots, SE {r['making_100_se']:.1f}",
                   size=s, fill="ink3")
            ly = ry + 12
        else:
            d.text(pad, ry + 12, f"{r['PLAYER_NAME']}: {r['fga']:,} shots, "
                   f"SE {r['making_100_se']:.1f}", size=s, weight="600")
            ly = ry + 56
        xo, xe = X(r["making_100"]), X(r["making_eb"])
        se = r["making_100_se"]
        d.line(X(r["making_100"] - se), ly, X(r["making_100"] + se), ly, stroke="ink3", sw=2)
        d.circle(xo, ly, 5.5, fill="none", stroke="c3", sw=2)
        d.line(xo - 7, ly + 16, xe + 7, ly + 16, stroke="c3", sw=1.8, arrow=True)
        d.circle(xe, ly, 5.5, fill="c3")
        d.line(X(mu), ly - 14, X(mu), ly + 22, stroke="ink2", sw=1.4, dash="4 3")
        keep = (f"keeps {r['weight']:.0%}: {r['making_100']:+.1f} "
                f"to {r['making_eb']:+.1f}")
        if wide:
            d.text(max(xo, xe) + 8, ly + 36, keep, size=s, fill="c3", anchor="end",
                   weight="600")
        else:
            d.text(pad, ry + 12 + s + 6, keep, size=s, fill="c3", weight="600")
    yb = top + len(rows) * rh + (6 if wide else -18)
    if wide:
        d.text(X(mu) + 4, top - 12, f"league mean {mu:+.1f}", size=s, fill="ink2")
    d.line(ax0, yb, ax1, yb, stroke="rule", sw=1)
    t = int(lo_v // 5 * 5 + 5)
    while t <= hi_v:
        d.line(X(t), yb, X(t), yb + 4, stroke="rule", sw=1)
        d.text(X(t), yb + s + 6, f"{t:+d}" if t else "0", size=s, fill="ink3",
               anchor="middle")
        t += 5
    d.text(ax1, yb + 2 * s + 12, "making, points above expected per 100 shots",
           size=s, fill="ink3", anchor="end")
    leg_y = yb + 2 * s + (34 if wide else 36)
    d.circle(pad + 6, leg_y - 4, 5.5, fill="none", stroke="c3", sw=2)
    d.text(pad + 18, leg_y, "observed, with 1 SE either side", size=s, fill="ink3")
    if wide:
        d.circle(pad + 300, leg_y - 4, 5.5, fill="c3")
        d.text(pad + 312, leg_y, "after shrinking", size=s, fill="ink3")
    else:
        d.circle(pad + 6, leg_y + 16, 5.5, fill="c3")
        d.text(pad + 18, leg_y + 20, "after shrinking", size=s, fill="ink3")
        d.line(pad + 6, leg_y + 28, pad + 6, leg_y + 44, stroke="ink2", sw=1.4, dash="4 3")
        d.text(pad + 18, leg_y + 40, f"league mean {mu:+.1f}", size=s, fill="ink3")
        leg_y += 40
    # step 3
    y3 = leg_y + (46 if wide else 44)
    d.text(pad, y3, "3. Project next season, then check", size=s + 2, weight="600")
    r = hiw
    calc = (f"{short(r['PLAYER_NAME'])}: diet {r['xpps']:.3f} plus shrunk making "
            f"{r['making_eb'] / 100:.3f} projects {r['proj_pps']:.3f} points per shot; "
            f"he scored {r['actual_pps_next']:.3f}.")
    y3 = lines(d, pad, y3 + s + 10, calc, W - 2 * pad, s, fill="ink2")
    res = (f"Across all {proj.height} players, the rule's average miss is "
           f"{gain:.1%} smaller than carrying raw points per shot forward.")
    end = lines(d, pad, y3 + s + 10, res, W - 2 * pad, s, weight="600")
    d.h = end + 16
    return d


# ---------------------------------------------------------------------------
# 3. End-of-period shots, split in two
# ---------------------------------------------------------------------------

def dg_clock_split(layout: str) -> fd.Diagram:
    c = pl.read_csv(OUT / "clock_decomposition.csv")
    base = c.filter(pl.col("bucket") == BASE).row(0, named=True)
    rlo, rhi = (int(v) for v in re.findall(r"\d+", RUSH)[:2])
    tlo, thi = (int(v) for v in re.findall(r"\d+", TFO)[:2])
    cols = [(f"Final {rhi} seconds", c.filter(pl.col("bucket") == RUSH).row(0, named=True)),
            (f"2-for-1 window, {tlo} to {thi} s",
             c.filter(pl.col("bucket") == TFO).row(0, named=True))]
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    rush = cols[0][1]
    title = (f"End-of-period shots split into selection and execution: the final "
             f"four seconds lose {per100(rush['xpps'] - base['xpps'])} and "
             f"{per100((rush['execution_100'] - base['execution_100']) / 100)} per 100")
    d = fd.Diagram(W, 0, title)
    d.text(pad, 26, "Points per shot, starting from normal play", size=s + 2,
           weight="600")
    lines(d, pad, 26 + s + 10, "Selection: the price of the shots taken. Execution: "
          "how they convert against that price.", W - 2 * pad, s, fill="ink3")
    lo_v, hi_v = 0.0, max(base["pps"], base["xpps"]) * 1.08
    if wide:
        panels = [(pad + 10, 96, 330, 300), (pad + 380, 96, 330, 300)]
    else:
        panels = [(pad, 96, W - 2 * pad, 300), (pad, 96 + 300 + 96, W - 2 * pad, 300)]
    for (name, r), (px, py, pw, ph) in zip(cols, panels):
        Y = lambda v, py=py, ph=ph: py + ph - (v - lo_v) / (hi_v - lo_v) * ph  # noqa: E731
        d.text(px, py - 6, f"{name}: {r['n']:,} shots", size=s, weight="600")
        bw = pw / 3 - 14
        steps = [("normal play", lo_v, base["pps"], "ink3"),
                 ("selection", r["xpps"] - base["xpps"] + base["pps"], base["pps"], "nba"),
                 ("execution", r["pps"], r["xpps"] - base["xpps"] + base["pps"], "c8")]
        for k, (lab, a, b, col) in enumerate(steps):
            bx = px + k * (pw / 3) + 7
            lo_y, hi_y = Y(min(a, b)), Y(max(a, b))
            d.rect(bx, hi_y, bw, max(lo_y - hi_y, 1.5), fill=col, rx=2,
                   opacity=0.85 if k else 0.5)
            if k == 0:
                val = f"{base['pps']:.3f}"
            else:
                val = per100(a - b)
            d.text(bx + bw / 2, hi_y - 6 if k == 0 else lo_y + s + 4, val, size=s,
                   fill="ink" if k == 0 else col, anchor="middle", weight="600")
            d.text(bx + bw / 2, py + ph + s + 8, lab, size=s, fill="ink3", anchor="middle")
            if k < 2:
                nxt = steps[k + 1]
                yy = Y(b if k == 0 else a)
                d.line(bx + bw, yy, bx + pw / 3 + 7, yy, stroke="ink3", sw=1, dash="3 3")
        d.text(px + pw - 4, Y(r["pps"]) + 2 * s + 14, f"ends at {r['pps']:.3f}",
               size=s, fill="ink2", anchor="end")
        d.line(px, py + ph, px + pw, py + ph, stroke="rule", sw=1)
    foot = "Selection and execution steps in points per 100 shots, against normal play."
    end = lines(d, pad, panels[-1][1] + panels[-1][3] + 2 * s + 30, foot, W - 2 * pad,
                s, fill="ink3")
    d.h = end + 14
    return d


# ---------------------------------------------------------------------------
# 4. The rulebook: where each league's three-point line sits
# ---------------------------------------------------------------------------

# Court geometry from each league's rulebook, in feet: drawn, never printed.
# NBA: arc 23 ft 9 in, corners 22 ft. WNBA (FIBA line): arc 6.75 m, corners
# 6.6 m. Basket centre 5.25 ft (63 in) from the baseline; court 50 ft wide.
LINES = {"NBA": (23.75, 22.0), "WNBA": (6.75 / 0.3048, 6.6 / 0.3048)}
RIM_Y, HALF_W = 5.25, 25.0
LEAGUE_ROW = {"NBA": "NBA 2024-25", "WNBA": "WNBA 2025"}


def half_court(d, x0, y0, k, arc, corner, col):
    """Half court from the baseline (bottom) up to DEPTH ft; k px per ft."""
    depth = 31.0
    X = lambda fx: x0 + fx * k  # noqa: E731
    Y = lambda fy: y0 + (depth - fy) * k  # noqa: E731
    d.rect(X(0), Y(depth), 2 * HALF_W * k, depth * k, fill="surface2", stroke="rule", sw=1)
    d.rect(X(HALF_W - 8), Y(19), 16 * k, 19 * k, fill="none", stroke="rule", sw=1)
    d.circle(X(HALF_W), Y(RIM_Y), 0.75 * k, fill="none", stroke="ink3", sw=1.4)
    d.line(X(HALF_W - 3), Y(4), X(HALF_W + 3), Y(4), stroke="ink3", sw=1.4)
    cy = RIM_Y + (arc ** 2 - corner ** 2) ** 0.5
    # corner zones: between the corner line and the sideline
    for xa, xb in ((0, HALF_W - corner), (HALF_W + corner, 2 * HALF_W)):
        d.rect(X(xa), Y(cy), (xb - xa) * k, cy * k, fill=col, opacity=0.35)
    path = (f"M{X(HALF_W - corner):.1f},{Y(0):.1f} L{X(HALF_W - corner):.1f},{Y(cy):.1f} "
            f"A{arc * k:.1f},{arc * k:.1f} 0 0 1 {X(HALF_W + corner):.1f},{Y(cy):.1f} "
            f"L{X(HALF_W + corner):.1f},{Y(0):.1f}")
    d.path(path, stroke=col, sw=2.4)
    return X, Y, cy


def dots(d, x0, y0, share, col, step):
    """100 dots, one per three-point attempt; the corner ones filled."""
    m = round(share * 100)
    for i in range(100):
        r, c = divmod(i, 10)
        d.circle(x0 + c * step, y0 + r * step, step * 0.34,
                 fill=col if i < m else "none", stroke=None if i < m else "rule",
                 sw=1)
    return m


def dg_corner_courts(layout: str) -> fd.Diagram:
    st = pl.read_csv(OUT / "wnba_league_structure.csv")
    rows = {lg: st.filter(pl.col("league") == name).row(0, named=True)
            for lg, name in LEAGUE_ROW.items()}
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    n, w_ = rows["NBA"], rows["WNBA"]
    title = (f"Corner threes are {n['corner_share_of_3s']:.1%} of NBA threes and "
             f"{w_['corner_share_of_3s']:.1%} of WNBA threes: the corner is only "
             f"closer in the NBA")
    d = fd.Diagram(W, 0, title)
    y = lines(d, pad, 26, "Where the three-point line sits, and what shooters do "
              "with it", W - 2 * pad, s + 2, weight="600")
    k = 6.0 if wide else 6.2
    panel_w = 2 * HALF_W * k
    y += 26
    end = y
    for i, (lg, col) in enumerate((("NBA", "nba"), ("WNBA", "wnba"))):
        r = rows[lg]
        arc, corner = LINES[lg]
        if wide:
            x0, y0 = pad + i * (panel_w + 60), y + 24
        else:
            x0, y0 = (W - panel_w) / 2, end + 24
        d.text(x0, y0 - 10, LEAGUE_ROW[lg], size=s + 1, weight="600", fill=col)
        X, Y, cy = half_court(d, x0, y0, k, arc, corner, col)
        note = (f"Corner threes are taken a median {r['corner_gap_ft']:.0f} ft "
                f"closer than arc threes.")
        ty = lines(d, x0, Y(0) + s + 12, note, panel_w, s, fill="ink2")
        step = 20 if wide else 18
        dy = ty + 22
        m = dots(d, x0 + step * 0.4, dy, r["corner_share_of_3s"], col, step)
        d.text(x0, dy + 10 * step + 4,
               f"Of every 100 threes, {m} come from a corner", size=s, weight="600")
        end = dy + 10 * step + 4 + (0 if wide else 30)
    foot = ("Shaded: the corner zones, where the line runs straight along the "
            "sideline. One dot per three-point attempt; filled dots are corner threes.")
    end = lines(d, pad, end + 2 * s + 6, foot, W - 2 * pad, s, fill="ink3")
    d.h = end + 14
    return d


BUILDERS = {"dg1_two_skills": dg_two_skills, "dg2_trust_diet": dg_trust_diet,
            "dg3_clock_split": dg_clock_split, "dg4_corner_courts": dg_corner_courts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fails = []
    for ref, build in BUILDERS.items():
        wide, narrow = build("wide"), build("narrow")
        if args.check:
            for dg, name in ((wide, f"{ref}.svg"), (narrow, f"{ref}_narrow.svg")):
                p = FIG / name
                if not p.exists() or p.read_text() != dg.svg():
                    fails.append(name)
        else:
            for p in fd.save(FIG, ref, wide, narrow):
                print("wrote", p.relative_to(ROOT))
    if args.check:
        for f in fails:
            print("FAIL", f, "differs from a fresh build")
        print("DIAGRAMS", "FAILED" if fails else f"PASSED: {len(BUILDERS)} diagrams")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
