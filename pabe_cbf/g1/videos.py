"""Comparison videos (English): PABe-CBF (B9) vs baselines on the simulated Unitree G1.

Each clip shows two methods side by side on the SAME human trajectory and seed. Episodes are chosen by an
explicit, reproducible criterion from the test set, and every clip shows the aggregate test statistic so the
example is not mistaken for the whole result.

Usage: python -m pabe_cbf.g1.videos --out results/g1
"""
import argparse
import os
import pickle
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import params as P
from ..core import method_from_spec
from ..scenarios import Participant, Scenario
from . import render as R
from .episode import G1Episode
from .geometry import G1_GEOM
from .model import RESULTS_DIR, check_results_dir
from .train_g1 import load

from matplotlib import get_data_path  # noqa: E402  (fuentes DejaVu incluidas con matplotlib: portables)
FONT = os.path.join(get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
FONTB = os.path.join(get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
F = lambda s, b=False: ImageFont.truetype(FONTB if b else FONT, s)
BG, INK, MUTED = (250, 250, 248), (20, 20, 20), (110, 110, 105)
BLUE, ORANGE, RED, GREEN, GRAY = (42, 120, 214), (235, 104, 52), (208, 59, 59), (12, 140, 70), (90, 90, 86)
PW, PH = 560, 420                 # render panel
CH = 120                          # chart strip
W = 2 * PW + 60
H = 70 + 34 + PH + 30 + CH + 64
MODE_EN = {"A": "Approach", "C": "Contact", "R": "Release / retreat"}
NAME_EN = {"B9": "PABe-CBF (proposed)", "B3": "HMM-entropy CBF (B3)", "A1": "PABe without viability governor (A1)",
           "B7": "SHIELD (B7)", "B1": "Fixed CBF (B1)"}
FPS, STEP = 33, 3                 # 3 control steps (30 ms) per frame -> real time


def camera(view="side"):
    c = mujoco.MjvCamera()
    c.lookat[:] = [0.40, 0.0, 0.98]
    c.lookat[:] = [0.38, -0.02, 0.97]
    c.distance, c.azimuth, c.elevation = 1.30, -112.0, -11.0
    return c


def run_rec(spec, k, models):
    sc = Scenario(k[0], Participant(k[1], geom=G1_GEOM), k[2])
    ep = G1Episode(method_from_spec(spec), sc, models, record=True)
    res, _, _ = ep.run()
    return ep.record, sc, res


def chart(draw, x0, y0, w, h, series, t_now, t_max, ylim, color, ref=None, ref_label="", ylabel=""):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(215, 214, 208))
    lo, hi = ylim
    sx = lambda t: x0 + 4 + (w - 8) * t / t_max
    sy = lambda v: y0 + h - 6 - (h - 14) * (min(max(v, lo), hi) - lo) / (hi - lo)
    if ref is not None:
        draw.line([x0 + 2, sy(ref), x0 + w - 2, sy(ref)], fill=(150, 150, 145), width=1)
        draw.text((x0 + w - 6, sy(ref) - 15), ref_label, font=F(12), fill=MUTED, anchor="ra")
    pts = [(sx(t), sy(v)) for t, v in series if t <= t_now]
    if len(pts) > 1:
        draw.line(pts, fill=color, width=3)
    draw.text((x0 + 6, y0 + 4), ylabel, font=F(13), fill=MUTED)


def panel(img, rec_i, series_i, x0, name, color, i, t_max, sig, event):
    d = ImageDraw.Draw(img)
    x = rec_i[min(i, len(rec_i) - 1)]
    d.text((x0 + PW / 2, 78), name, font=F(20, True), fill=color, anchor="ma")
    img.paste(Image.fromarray(series_i["frames"][i // STEP]), (x0, 104))
    fy = 104 + PH + 4
    Fv = abs(x["F"])
    d.text((x0 + 4, fy), f"Mode: {MODE_EN[x['mode']]}", font=F(15), fill=INK)
    d.text((x0 + PW - 4, fy), f"|F| = {Fv:4.1f} N", font=F(15, True), fill=RED if Fv > P.F_PHYS else INK, anchor="ra")
    d.text((x0 + PW / 2, fy), f"margin θd = {100 * x['theta'][0]:.1f} cm", font=F(15), fill=INK, anchor="ma")
    if not x.get("feasible", True):
        d.rectangle([x0 + 8, 112, x0 + 212, 140], fill=RED)
        d.text((x0 + 14, 116), "QP INFEASIBLE → backup", font=F(15, True), fill=(255, 255, 255))
    if event:
        txt, col = event
        tw = d.textlength(txt, font=F(19, True))
        d.rectangle([x0 + PW / 2 - tw / 2 - 12, 104 + PH - 46, x0 + PW / 2 + tw / 2 + 12, 104 + PH - 12], fill=col)
        d.text((x0 + PW / 2, 104 + PH - 42), txt, font=F(19, True), fill=(255, 255, 255), anchor="ma")
    cy = 104 + PH + 30
    chart(d, x0, cy, PW, CH, series_i["series"], x["t"], t_max, sig["ylim"], color, sig.get("ref"),
          sig.get("ref_label", ""), sig["ylabel"])
    k_inf = series_i["cum_inf"][min(i, len(series_i["cum_inf"]) - 1)]
    d.text((x0 + PW - 6, cy + 4), f"infeasible-QP cycles: {k_inf}", font=F(13, True),
           fill=RED if k_inf else MUTED, anchor="ra")


def title_card(lines, path, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H / 2 - 30 * len(lines)
    for j, (txt, size, bold, col) in enumerate(lines):
        d.text((W / 2, y), txt, font=F(size, bold), fill=col, anchor="ma")
        y += size + 22
    if sub:
        d.text((W / 2, H - 50), sub, font=F(14), fill=MUTED, anchor="ma")
    img.save(path)


def make_clip(tag, title, subtitle, specs, names, colors, k, models, sig_fn, sig, stat_line, out_dir, criterion):
    recs = []
    for s in specs:
        rec, sc, res = run_rec(s, k, models)
        recs.append((rec, res))
    n = max(len(r[0]) for r in recs)          # el clip dura lo que el episodio más largo
    idx = list(range(0, n, STEP))
    t_max = (n - 1) * 0.01
    data = []
    for (rec, res), s in zip(recs, specs):
        frames = R.frames(rec, [min(i, len(rec) - 1) for i in idx], PW, PH)   # congela el último fotograma
        series = [(x["t"], sig_fn(x, s)) for x in rec]
        cum_inf = np.cumsum([0 if x.get("feasible", True) else 1 for x in rec]).tolist()
        data.append(dict(frames=frames, series=series, cum_inf=cum_inf))
    # event banners from mode transitions and outcomes
    evs = []
    for (rec, res), s in zip(recs, specs):
        ev, done_close = {}, False
        f_over = False
        for i in range(1, len(rec)):
            a, b = rec[i - 1], rec[i]
            lab = None
            if a["mode"] == "C" and b["mode"] == "R":
                if res["t_closure"] is not None and abs(b["t"] - res["t_closure"]) < 0.02:
                    lab = ("Handshake completed ✓", GREEN)
                elif res["withdraw"] and res["t_W"] is not None and b["t"] >= res["t_W"]:
                    lab = ("Released: person withdrew", BLUE if s.startswith("B9") else GRAY)
                else:
                    why = {"sem": "risk rule fired", "backup": "QP infeasible → backup", "pull": "pull detected",
                           "hw": "hardware force trip"}.get(res["R_reason"], res["R_reason"])
                    lab = (f"FALSE RELEASE ({why})", RED)
            if a["mode"] == "A" and b["mode"] == "R" and not res["withdraw"]:
                lab = ("ABORTED handshake (false cancellation)", RED)
            if abs(b["F"]) > P.F_PHYS and not f_over:
                f_over = True
                lab = (f"FORCE LIMIT EXCEEDED (> {P.F_PHYS:.0f} N)", RED)
            if lab:
                for j in range(i, n):              # el resultado permanece en pantalla
                    ev[j] = lab

        evs.append(ev)
    tmp = os.path.join(out_dir, f"_frames_{tag}")
    os.makedirs(tmp, exist_ok=True)
    for j, i in enumerate(idx):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((W / 2, 12), title, font=F(24, True), fill=INK, anchor="ma")
        d.text((W / 2, 44), subtitle, font=F(15), fill=MUTED, anchor="ma")
        for p_, (dd, nm, col) in enumerate(zip(data, names, colors)):
            x0 = 20 + p_ * (PW + 20)
            panel(img, recs[p_][0], dd, x0, nm, col, i, t_max, sig, evs[p_].get(i))
        d.text((W / 2, H - 56), f"t = {i * 0.01:.2f} s   ·   same person, same trajectory, same random seed   ·   real time",
               font=F(13), fill=MUTED, anchor="ma")
        d.text((W / 2, H - 34), stat_line, font=F(15, True), fill=INK, anchor="ma")
        img.save(os.path.join(tmp, f"{j + 100:05d}.png"))
    # title card (held 2.4 s) + final frame hold (1.5 s)
    title_card([(title, 30, True, INK), (subtitle, 18, False, MUTED), (stat_line, 18, True, BLUE)],
               os.path.join(tmp, "card.png"), sub="Selection criterion: " + criterion)
    for j in range(100):
        os.link(os.path.join(tmp, "card.png"), os.path.join(tmp, f"{j:05d}.png"))
    os.remove(os.path.join(tmp, "card.png"))
    last = os.path.join(tmp, f"{len(idx) + 99:05d}.png")
    for j in range(66):
        p = os.path.join(tmp, f"{len(idx) + 100 + j:05d}.png")
        if not os.path.exists(p):
            os.link(last, p)
    out = os.path.join(out_dir, f"{tag}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    for f_ in os.listdir(tmp):
        os.remove(os.path.join(tmp, f_))
    os.rmdir(tmp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RESULTS_DIR)
    a = ap.parse_args()
    check_results_dir(a.out)
    R._camera = camera
    models = load(os.path.join(a.out, "models.pkl"))
    T = pickle.load(open(os.path.join(a.out, "test.pkl"), "rb"))
    rows = {}
    for r, _ in T["out"]:
        rows.setdefault(r["spec"], []).append(r)
    sp = {s.split("@")[0]: s for s in T["specs"]}
    key = lambda r: (r["family"], r["pid"], r["seed"])
    I = {b: {key(r): r for r in rows[sp[b]]} for b in ("B9", "B3", "A1", "B7")}
    pct = lambda b, f, m: 100 * np.nanmean([float(r[m]) for r in rows[sp[b]] if f is None or r["family"] == f])
    vdir = os.path.join(a.out, "paper", "videos")
    os.makedirs(vdir, exist_ok=True)
    tau3 = float(sp["B3"].split("@")[1]); tau9 = float(sp["B9"].split("@")[1])

    # 1) benign hesitation: B3 false release, B9 completes (first such participant without domain shift)
    c1 = sorted(k for k, r in I["B9"].items() if k[0] == "iv_hesitation" and r["success"] is True
                and I["B3"][k]["false_cancel"] is True and I["B3"][k]["R_reason"] in ("sem", "abort") and not r["shift"])
    k1 = c1[0]
    s1 = make_clip("1_hesitation_B9_vs_B3", "Benign hesitation during the handshake",
                   "The person hesitates but does NOT withdraw — releasing is a false cancellation",
                   [sp["B9"], sp["B3"]], [NAME_EN["B9"], NAME_EN["B3"]], [BLUE, ORANGE], k1, models,
                   lambda x, s: (x["p"] / tau9 if s.startswith("B9") else float(x["b"][2] + 0.4 * x["H"]) / tau3),
                   dict(ylim=(0, 2.5), ylabel="release score / own threshold", ref=1.0, ref_label="release if > 1"),
                   f"Test set, hesitation scenario (n=40 each): false cancellations  PABe-CBF {pct('B9', 'iv_hesitation', 'false_cancel'):.0f}%  vs  B3 {pct('B3', 'iv_hesitation', 'false_cancel'):.0f}%",
                   vdir, "first test participant (no domain shift) where B3's risk rule fires falsely and PABe-CBF completes")
    print(s1)

    # 2) withdrawal: A1 (instant margins) loses feasibility, B9 does not; largest jerk gap, no violations
    c2 = sorted([k for k, r in I["B9"].items() if k[0] == "iii_withdraw_contact" and not r["shift"]
                 and I["A1"][k]["n_infeas"] > 0 and r["n_infeas"] == 0 and not r["V_any"] and not I["A1"][k]["V_any"]],
                key=lambda k: -(I["A1"][k]["jerk_p95"] - I["B9"][k]["jerk_p95"]))
    k2 = c2[0]
    s2 = make_clip("2_viability_B9_vs_A1", "Viability governor: smooth safety margins",
                   "Same risk signal; A1 jumps the margin instantly, PABe-CBF lets the QP decide how fast it grows",
                   [sp["B9"], sp["A1"]], [NAME_EN["B9"], NAME_EN["A1"]], [BLUE, GRAY], k2, models,
                   lambda x, s: 100 * x["theta"][0],
                   dict(ylim=(0, 10), ylabel="safety margin θd (cm)"),
                   f"Test set (n=360 each): episodes with infeasible QP  PABe-CBF {100 * np.mean([r['n_infeas'] > 0 for r in rows[sp['B9']]]):.0f}%  vs  A1 {100 * np.mean([r['n_infeas'] > 0 for r in rows[sp['A1']]]):.0f}%",
                   vdir, "withdrawal episode (no domain shift) with the largest jerk gap where A1 loses feasibility and PABe-CBF does not")
    print(s2)

    # 3) fast withdrawal in contact: SHIELD exceeds the force limit, B9 does not (B9 must also have grasped)
    c3 = sorted([k for k, r in I["B9"].items() if k[0] == "iii_withdraw_contact" and not r["shift"]
                 and I["B7"][k]["V_force"] and not r["V_any"] and r["release_applicable"]],
                key=lambda k: -I["B7"][k]["peak_F"])
    if not c3:
        c3 = sorted([k for k, r in I["B9"].items() if k[0] == "iii_withdraw_contact" and not r["shift"]
                     and I["B7"][k]["V_force"] and not r["V_any"]], key=lambda k: -I["B7"][k]["peak_F"])
    k3 = c3[0]
    s3 = make_clip("3_withdrawal_B9_vs_SHIELD", "Fast withdrawal while holding hands",
                   "The person pulls the hand away; the robot must release before the force limit (20 N)",
                   [sp["B9"], sp["B7"]], [NAME_EN["B9"], NAME_EN["B7"]], [BLUE, ORANGE], k3, models,
                   lambda x, s: abs(x["F"]),
                   dict(ylim=(0, 32), ylabel="interaction force |F| (N)", ref=P.F_PHYS, ref_label="20 N limit"),
                   f"Test set, withdrawal-in-contact scenario: force-limit violations  PABe-CBF {pct('B9', 'iii_withdraw_contact', 'V_force'):.0f}%  vs  SHIELD {pct('B7', 'iii_withdraw_contact', 'V_force'):.0f}%",
                   vdir, "withdrawal episode (no domain shift) where SHIELD exceeds the force limit and PABe-CBF does not; highest SHIELD peak force")
    print(s3)

    # combined video: intro + 3 clips + summary
    title_card([("PABe-CBF vs. baselines", 38, True, INK), ("Unitree G1 handshake · MuJoCo simulation", 22, False, MUTED),
                ("1) Benign hesitation   2) Viability governor   3) Fast withdrawal", 18, False, INK)],
               os.path.join(vdir, "_intro.png"), sub="All methods share the same nominal controller, sensors, physical limits and backup policy.")
    S = lambda b: rows[sp[b]]
    title_card([("Summary over the full test set (360 episodes per method)", 26, True, INK),
                (f"Physical violations: PABe-CBF {100 * np.mean([r['V_any'] for r in S('B9')]):.0f}%  ·  fixed CBF {100 * np.mean([r['V_any'] for r in S('B1')]):.1f}%  ·  SHIELD {100 * np.mean([r['V_any'] for r in S('B7')]):.0f}%", 20, False, INK),
                (f"Infeasible-QP episodes: PABe-CBF {100 * np.mean([r['n_infeas'] > 0 for r in S('B9')]):.0f}%  vs  no governor {100 * np.mean([r['n_infeas'] > 0 for r in S('A1')]):.0f}%", 20, False, INK),
                (f"False cancellations: PABe-CBF {100 * np.nanmean([r['false_cancel'] for r in S('B9')]):.0f}%  vs  entropy rule B3 {100 * np.nanmean([r['false_cancel'] for r in S('B3')]):.0f}%", 20, False, INK),
                ("Limitation: completed handshakes 64% (fixed CBF 70%) — safety costs some utility.", 17, False, MUTED)],
               os.path.join(vdir, "_outro.png"), sub="Simulation results; not validated on hardware.")
    for nm in ("_intro", "_outro"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", "4", "-framerate", str(FPS), "-i",
                        os.path.join(vdir, nm + ".png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        os.path.join(vdir, nm + ".mp4")], check=True)
    lst = os.path.join(vdir, "_list.txt")
    with open(lst, "w") as f:
        for p in ["_intro.mp4", os.path.basename(s1), os.path.basename(s2), os.path.basename(s3), "_outro.mp4"]:
            f.write(f"file '{p}'\n")
    allv = os.path.join(vdir, "PABe-CBF_vs_baselines_all.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", allv], check=True)
    for nm in ("_intro.mp4", "_outro.mp4", "_intro.png", "_outro.png", "_list.txt"):
        os.remove(os.path.join(vdir, nm))
    print(allv)
    with open(os.path.join(vdir, "episodes.txt"), "w") as f:
        f.write(f"clip1 {k1}\nclip2 {k2}\nclip3 {k3}\n")


if __name__ == "__main__":
    main()
