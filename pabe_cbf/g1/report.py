"""Figuras, tablas LaTeX y cifras del banco G1 a partir de results/g1/*.pkl.

Uso: python -m pabe_cbf.g1.report --out results/g1
"""
import argparse
import json
import os
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import params as P
from ..core import method_from_spec
from ..metrics import rate, cluster_paired_bootstrap, holm, prediction_metrics
from ..paper_figs import style, save, fnum, fp, fpct, tex_table, INK, INK2, MUTED, GRID, EDGE, BLUE, ORANGE, AQUA, \
    reliability, COL1, COL2
from ..scenarios import FAMILIES_G1, Participant, Scenario
from .experiments import detected, controlled_cells, STRESS, TEST0
from .geometry import G1_GEOM

FAM_ES = {"i_stable": "i estable", "ii_withdraw_pre": "ii retirada pre", "iii_withdraw_contact": "iii retirada contacto",
          "iv_hesitation": "iv duda inocua", "v_occlusion": "v oclusión", "vi_perturbation": "vi perturbación",
          "vii_lunge": "vii embestida", "viii_ambiguous_withdraw": "viii retirada ambigua", "ix_reactive": "ix reactivo"}
NAME = {"B0": "B0 nominal", "B1": "B1 CBF fijo", "B2": "B2 HMM-MAP", "B3": "B3 HMM-$H(b)$", "B4": "B4 BCBF",
        "B5": "B5 CRC-CBF", "B6": "B6 TV-CBF", "B7": "B7 SHIELD", "B8": "B8 RCIP", "B9": "B9 PABe-CBF",
        "A1": "A1 sin gobernador", "A2": "A2 $H(b)$ global", "A3": "A3 sin $p_W^+$", "A4": "A4 $b_W$ crudo",
        "A5": "A5 barrera única", "A6": "A6 HSMM", "A7": "A7 sin táctil", "A8": "A8 $p_W^+$ bins",
        "A9": "A9 $\\rho\\to\\infty$", "A10": "A10 secuencial", "A11": "A11 regla $p+0{,}4H$"}
STRESS_ES = {"noise_scale": "ruido de posición (x)", "vis_delay": "retardo visual extra (s)",
             "obs_noise": "ruido de observación HMM (x)", "ext_force": "empujón externo (N)",
             "tactile_delay": "retardo táctil (s)", "lunge_amp": "amplitud de embestida (m)"}


def base(s):
    return s.split("@")[0]


def lab_tex(s):
    return lab(s).replace("κ", "$\\kappa$").replace("∞", "$\\infty$")


def lab(s):
    p = s.split("@")
    extra = []
    for q in p[1:]:
        if q.startswith("k"):
            extra.append("κ=" + ("∞" if q == "kinf" else q[1:]))
    return NAME[p[0]] + (f" ({', '.join(extra)})" if extra else "")


def key(r):
    return (r["family"], r["pid"], r["seed"])


def summarize_g1(rows, extras=None):
    col = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], dtype=float)
    s = {}
    s["V_any"] = rate(col("V_any"))
    for k in ("dist_link", "dist_palm_body", "force", "unauth_contact"):
        s[f"V_{k}"] = rate(col(f"V_{k}"))
    s["peak_F_p95"] = float(np.percentile(col("peak_F"), 95))
    s["peak_F_mean"] = float(col("peak_F").mean())
    rel = [r for r in rows if r["release_applicable"]]
    s["release_fail"] = rate([r["release_fail"] for r in rel])
    rdt = np.array([r["release_dt"] for r in rel], float) * 1e3
    s["release_ms_median"] = float(np.nanmedian(rdt)) if np.isfinite(rdt).any() else np.nan
    s["false_cancel"] = rate(col("false_cancel"))
    s["success_auth"] = rate([r["success"] for r in rows if r["authorized"] and not r["withdraw"]])
    s["sens"] = rate([detected(r) for r in rows if r["withdraw"]])
    s["infeas_ep"] = rate(col("n_infeas") > 0)
    s["infeas_cycles"] = float(col("n_infeas").sum() / max(col("n_qp").sum(), 1))
    s["jerk"] = float(col("jerk_p95").mean())
    s["J_int"] = float(col("J_int").mean())
    s["gov_frac"] = float(col("gov_frac").mean())
    s["iv_n"] = float(col("n_iv").mean())
    ivs = col("iv_max")
    s["iv_max_p95"] = float(np.percentile(ivs[ivs > 0], 95)) if (ivs > 0).any() else 0.0
    s["iv_unrec"] = rate([r["iv_unrecovered"] for r in rows if r["n_iv"] > 0])
    s["iv_viol"] = rate([r["iv_viol"] for r in rows if r["n_iv"] > 0])
    s["reacted"] = rate([r["reacted"] for r in rows if r["family"] == "ix_reactive"])
    if extras:
        ts = np.concatenate([e["t_safety"] for e in extras])
        s["lat"] = [float(np.percentile(ts, q)) for q in (50, 95, 99)] + [float(ts.max())]
        tq = np.concatenate([e["t_qp"] for e in extras]) if any(len(e["t_qp"]) for e in extras) else np.zeros(1)
        s["qp"] = [float(np.percentile(tq, 50)), float(np.percentile(tq, 99))]
        th = np.concatenate([e["t_hmm"] for e in extras])
        s["hmm50"] = float(np.percentile(th, 50)) if len(th) else np.nan
        s["over_budget"] = float((ts > P.LATENCY_BUDGET_MS).mean())
    return s


def contrast(rows_a, rows_b, f, filt=None):
    ia, ib = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    va, vb, cl = [], [], []
    for k in sorted(set(ia) & set(ib)):
        if filt and not filt(ia[k]):
            continue
        va.append(f(ia[k])); vb.append(f(ib[k])); cl.append(k[1])
    return cluster_paired_bootstrap(va, vb, cl)


F_BIN = lambda k: (lambda r: np.nan if r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])) else float(r[k]))


# ======================================================================================
def fig_scene(T, models, out, sp):
    """Escenas renderizadas (B1 vs B9) en un episodio de retirada durante el contacto."""
    from PIL import Image
    from . import render as R
    from .episode import G1Episode
    import mujoco

    def cam(view="side"):
        c = mujoco.MjvCamera(); c.lookat[:] = [0.40, 0.0, 0.98]
        c.distance, c.azimuth, c.elevation = 1.6, -110.0, -10.0
        return c
    R._camera = cam
    rows1 = {key(r): r for r in T[sp["B1"]]}
    best, bk = -1, None
    for r in T[sp["B9"]]:
        k = key(r)
        if r["family"] == "iii_withdraw_contact" and k in rows1 and r["release_applicable"] and not r["shift"] \
                and rows1[k]["release_applicable"]:
            dlt = rows1[k]["peak_F"] - r["peak_F"]
            if dlt > best:
                best, bk = dlt, k
    if bk is None:      # respaldo: primer episodio de retirada en contacto disponible
        bk = next(key(r) for r in T[sp["B9"]] if r["family"] == "iii_withdraw_contact")
    recs = {}
    for name in ("B1", "B9"):
        ep = G1Episode(method_from_spec(sp[name]), Scenario(bk[0], Participant(bk[1], geom=G1_GEOM), bk[2]), models,
                       record=True)
        ep.run(); recs[name] = (ep.record, ep.sc)
    rec9, sc = recs["B9"]
    t_C = next((x["t"] for x in rec9 if x["mode"] == "C"), 1.0)
    t_eng = next((x["t"] for x in rec9 if x["engaged"]), t_C + 0.3)
    t_cont = max(t_eng + 0.15, sc.t_W - 0.15) if t_eng < sc.t_W else t_eng
    times = [max(0.0, t_C - 0.3), t_cont, sc.t_W + 0.25, sc.t_W + 0.55]
    names = ["aproximación", "contacto (antes de retirarse)", "retirada +0,25 s", "retirada +0,55 s"]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.9))
    for row, name in enumerate(("B1", "B9")):
        rec = recs[name][0]
        idx = [min(int(round(t / 0.01)), len(rec) - 1) for t in times]
        imgs = R.frames(rec, idx, 520, 420)
        for c, (im, i) in enumerate(zip(imgs, idx)):
            ax = axes[row, c]
            ax.imshow(im[30:400, 60:470]); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            x = rec[i]
            if row == 0:
                ax.set_title(f"{names[c]}\nt = {x['t']:.2f} s", fontsize=7)
            ax.text(0.5, -0.03, f"modo {x['mode']} · |F| = {abs(x['F']):.1f} N\n$\\theta_d$ = {100 * x['theta'][0]:.1f} cm",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=INK2)
        axes[row, 0].text(-0.05, 0.5, NAME[name], transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right",
                          fontsize=7.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.38)
    save(fig, out, "g1_scene")
    try:
        R.video([recs["B1"][0], recs["B9"][0]], ["B1 CBF fijo", "B9 PABe-CBF"], os.path.join(out, "figures", "g1_B1_vs_B9.mp4"))
    except Exception as e:      # el vídeo es opcional
        print("[report] vídeo no generado:", e)
    return bk, recs


def fig_timeseries(models, out, bk, sp, recs):
    from .episode import G1Episode
    for name in ("A1", "B3"):
        ep = G1Episode(method_from_spec(sp[name]), Scenario(bk[0], Participant(bk[1], geom=G1_GEOM), bk[2]), models,
                       record=True)
        ep.run(); recs[name] = (ep.record, ep.sc)
    r9, sc = recs["B9"]
    t9 = np.array([x["t"] for x in r9])
    fig, axes = plt.subplots(5, 1, figsize=(COL2, 6.0), sharex=True)
    shade = {"C": "#f2f1ec", "R": "#e7eef9"}
    for ax in axes:
        m0 = 0
        for i in range(1, len(r9) + 1):
            if i == len(r9) or r9[i]["mode"] != r9[m0]["mode"]:
                if r9[m0]["mode"] in shade:
                    ax.axvspan(t9[m0], t9[i - 1], color=shade[r9[m0]["mode"]], lw=0, zorder=0)
                if ax is axes[0]:
                    ax.text((t9[m0] + t9[i - 1]) / 2, 1.03, r9[m0]["mode"], transform=ax.get_xaxis_transform(),
                            ha="center", fontsize=6.5, color=INK2)
                m0 = i
        ax.axvline(sc.t_W, color=ORANGE, lw=0.8, ls=(0, (3, 2)))
        for x in r9:
            if not x["feasible"]:
                ax.axvline(x["t"], color="#d03b3b", lw=0.4, alpha=0.5, zorder=1)
    B = np.array([x["b"] for x in r9])
    for z, c, n in ((0, BLUE, "E"), (1, AQUA, "H"), (2, ORANGE, "W")):
        axes[0].plot(t9, B[:, z], color=c, label=n)
    axes[0].set_ylabel("creencia"); axes[0].legend(ncol=3, loc="center left", fontsize=6.5)
    tau = float(sp["B9"].split("@")[1])
    axes[1].plot(t9, [x["p"] for x in r9], color=BLUE, label="$p_W^+$")
    axes[1].plot(t9, [x["H"] for x in r9], color=MUTED, lw=1, label="$H_t$")
    axes[1].plot(t9, [x["Eexc"] for x in r9], color=AQUA, label="$E^{exc}_t$")
    axes[1].axhline(tau, color=BLUE, lw=0.6, ls=(0, (4, 2)))
    axes[1].set_ylabel("señal"); axes[1].legend(ncol=3, loc="center left", fontsize=6.5)
    axes[2].plot(t9, [100 * x["theta_star"][0] for x in r9], color=MUTED, lw=0.9, ls=(0, (2, 1.5)), label="$\\theta_d^\\star$")
    axes[2].plot(t9, [100 * x["theta"][0] for x in r9], color=BLUE, label="B9 (gobernador)")
    ra = recs["A1"][0]
    axes[2].plot([x["t"] for x in ra], [100 * x["theta"][0] for x in ra], color=INK2, lw=0.9, label="A1 (instantáneo)")
    axes[2].set_ylabel("$\\theta_d$ (cm)"); axes[2].legend(ncol=3, loc="upper left", fontsize=6.5)
    for name, c, lw in (("B1", INK2, 1.0), ("B3", ORANGE, 1.0), ("B9", BLUE, 1.4)):
        rr = recs[name][0]
        axes[3].plot([x["t"] for x in rr], [abs(x["F"]) for x in rr], color=c, lw=lw, label=NAME[name])
    axes[3].axhline(P.F_PHYS, color=INK, lw=0.7)
    axes[3].text(t9[0], P.F_PHYS, " $F_{phys}$", fontsize=6, va="bottom")
    axes[3].set_ylabel("|F| (N)"); axes[3].legend(ncol=3, loc="center left", fontsize=6.5)
    axes[3].set_ylim(-1, max(P.F_PHYS + 6, axes[3].get_ylim()[1]))
    for name, c, ls in (("B1", INK2, "-"), ("B3", ORANGE, "-"), ("A1", MUTED, (0, (2, 1))), ("B9", BLUE, "-")):
        rr = recs[name][0]
        axes[4].plot([x["t"] for x in rr], [np.linalg.norm(x["u"] - x["u_nom"]) for x in rr], color=c, ls=ls,
                     lw=1.3 if name == "B9" else 0.9, label=NAME[name])
    axes[4].set_ylabel("$\\|u-u_{nom}\\|$"); axes[4].set_xlabel("tiempo (s)")
    axes[4].legend(ncol=4, loc="upper left", fontsize=6.5)
    for i, ax in enumerate(axes):
        ax.text(-0.09, 1.0, f"({'abcde'[i]})", transform=ax.transAxes, fontsize=8, va="top")
    axes[-1].set_xlim(t9[0], t9[-1])
    fig.align_ylabels(axes)
    save(fig, out, "g1_timeseries")


def fig_heatmaps(T, specs, out):
    mets = [("V_any", "violación física (%)"), ("false_cancel", "cancelación falsa (%)"),
            ("release_fail", "liberación fallida (%)"), ("infeas", "episodios con QP infactible (%)")]
    fams = list(FAMILIES_G1)
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 3.6), sharey=True)
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#f6f9fe", "#9ec5f4", "#2a78d6", "#0d366b"])
    tabs = {}
    for ax, (mk, title) in zip(axes, mets):
        M = np.full((len(specs), len(fams)), np.nan)
        for i, s in enumerate(specs):
            for j, f in enumerate(fams):
                rs = [r for r in T[s] if r["family"] == f]
                if mk == "infeas":
                    v = [r["n_infeas"] > 0 for r in rs]
                elif mk == "release_fail":
                    v = [r["release_fail"] for r in rs if r["release_applicable"]]
                else:
                    v = [r[mk] for r in rs]
                v = [x for x in v if x == x and x is not None]
                if v:
                    M[i, j] = 100 * np.mean(v)
        tabs[mk] = M
        ax.imshow(np.nan_to_num(M, nan=0), cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=4.8,
                            color="white" if M[i, j] > 55 else INK)
                else:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=5, color=MUTED)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAM_ES[f].split()[0] for f in fams], fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axes[0].set_yticks(range(len(specs)))
    axes[0].set_yticklabels([NAME[base(s)] for s in specs], fontsize=6)
    fig.text(0.5, -0.02, "familias: " + "; ".join(FAM_ES[f] for f in fams) + ". «·»: no aplica.", ha="center",
             fontsize=5.8, color=INK2)
    save(fig, out, "g1_scenarios")
    return tabs


def fig_frontier(V, T, out, specs):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True)
    cols = {"B9": BLUE, "B3": ORANGE, "B5": AQUA}
    ch = V["chosen"]
    for ax, ttl, src in ((axes[0], "validación (selección de umbral)", "val"), (axes[1], "test", "test")):
        for b, pts in V["front"].items():
            if b not in ("B3", "B4", "B5", "B8", "B9", "A6", "A8"):
                continue
            if src == "val":
                P_ = [(p["spec"], 100 * p["sens"]["rate"], 100 * p["fc"]["rate"]) for p in pts]
            else:
                P_ = []
                for p in pts:
                    if p["spec"] in T:
                        s = summarize_g1(T[p["spec"]])
                        P_.append((p["spec"], 100 * s["sens"]["rate"], 100 * s["false_cancel"]["rate"]))
            if not P_:
                continue
            P_.sort(key=lambda x: x[1])
            c = cols.get(b, MUTED)
            ax.plot([x[1] for x in P_], [x[2] for x in P_], "-o", color=c, ms=3, lw=1.3 if b in cols else 0.8,
                    mec="white", mew=0.5, alpha=1 if b in cols else 0.8)
            ax.annotate(b, (P_[-1][1], P_[-1][2]), textcoords="offset points", xytext=(3, 0), fontsize=6, color=c)
            for x in P_:
                if x[0] == ch.get(b):
                    ax.plot(x[1], x[2], "o", ms=7, mfc="none", mec=c, mew=1.1)
        ax.axvline(100 * V["s_star"], color=MUTED, lw=0.7, ls=(0, (2, 2)))
        ax.set_title(ttl); ax.set_xlabel("sensibilidad de retirada (%)")
    axes[0].set_ylabel("cancelación falsa (%)")
    axes[1].text(0.02, 0.97, "en test sólo existen los puntos elegidos\n(los barridos completos se evaluaron en validación)",
                 transform=axes[1].transAxes, fontsize=5.8, color=INK2, va="top")
    save(fig, out, "g1_frontier")


def fig_attribution(T, out, sp, extra):
    """(a) Pareto infactibilidad-suavidad: ¿puede alguna ley exógena (B6, kappa) igualar al gobernador?
    (b) efecto B9-B6* estratificado por si la viabilidad estuvo comprometida en el episodio (B6* infactible)."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.75))
    ax = axes[0]
    ks = [s for s in extra + [sp["B6"]] if base(s) == "B6"]
    ks = sorted(set(ks), key=lambda s: float("inf") if s.endswith("kinf") else float(s.split("@k")[1]))
    pts = []
    for s in ks:
        S = summarize_g1(T[s])
        pts.append((s, 100 * S["infeas_ep"]["rate"], S["jerk"]))
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "-o", color=MUTED, ms=3.5, lw=1)
    for s, x, y in pts:
        ax.annotate(lab(s).split("(")[-1].rstrip(")"), (x, y), textcoords="offset points", xytext=(3, 2), fontsize=5.8,
                    color=INK2)
    for name, c, mk in (("A1", INK2, "s"), ("A10", INK2, "^"), ("A9", AQUA, "D"), ("B9", BLUE, "o")):
        S = summarize_g1(T[sp[name]])
        x, y = 100 * S["infeas_ep"]["rate"], S["jerk"]
        ax.plot(x, y, mk, color=c, ms=6 if name == "B9" else 4.5, mec="white", mew=0.6)
        ax.annotate(NAME[name].split()[0], (x, y), textcoords="offset points", xytext=(4, -8), fontsize=6, color=c)
    ax.set_xlabel("episodios con QP infactible (%)"); ax.set_ylabel("jerk p95 de la palma (m/s³)")
    ax.set_title("(a) leyes exógenas de θ (B6, κ) frente al gobernador", fontsize=7)
    # (b) estratificación + contraste puro A9 vs B6(kappa=4) (misma referencia; sólo difieren si es inviable)
    axes[1].remove()
    gs = axes[0].get_gridspec()
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.08)
    b6 = {key(r): r for r in T[sp["B6"]]}
    groups = [("todos", None), ("B6* sin infact.", False), ("B6* con infact.", True)]
    res = {}
    mets = (("jerk p95 (m/s³)", lambda r: r["jerk_p95"], 1), ("QP infactible (pp)", lambda r: float(r["n_infeas"] > 0), 100),
            ("violación (pp)", lambda r: float(r["V_any"]), 100))
    for j_, (mn, f, sc_) in enumerate(mets):
        axb = fig.add_subplot(sub[0, j_])
        ylab = []
        for i_, (gn, cond) in enumerate(groups):
            ks_ = {key(r) for r in T[sp["B9"]] if key(r) in b6 and (cond is None or (b6[key(r)]["n_infeas"] > 0) == cond)}
            c = contrast([r for r in T[sp["B9"]] if key(r) in ks_], [r for r in T[sp["B6"]] if key(r) in ks_], f)
            res[(mn, gn)] = c
            y = 3 - i_
            col = BLUE
            axb.plot([sc_ * c["lo"], sc_ * c["hi"]], [y, y], color=col, lw=1.1)
            axb.plot(sc_ * c["diff"], y, "o", color=col, ms=3.5)
            ylab.append((y, f"B9−B6*: {gn} (n={c['n']})"))
        c = contrast(T[sp["A9"]], T[sp["B6ref"]], f)
        res[(mn, "A9 vs B6(κ=4)")] = c
        axb.plot([sc_ * c["lo"], sc_ * c["hi"]], [0, 0], color=AQUA, lw=1.1)
        axb.plot(sc_ * c["diff"], 0, "D", color=AQUA, ms=3.5)
        ylab.append((0, f"A9−B6(κ=4) (n={c['n']})"))
        axb.axvline(0, color=INK, lw=0.6)
        axb.set_ylim(-0.6, 3.6)
        axb.set_title(mn, fontsize=6.5)
        axb.tick_params(labelsize=5.5)
        if j_ == 0:
            axb.set_yticks([y for y, _ in ylab]); axb.set_yticklabels([t for _, t in ylab], fontsize=5.5)
        else:
            axb.set_yticks([])
        axb.grid(axis="y", visible=False)
    fig.text(0.74, 1.0, "(b) diferencias pareadas (IC 95 % por participante)", ha="center", fontsize=7)
    save(fig, out, "g1_attribution")
    return pts, res


def fig_sensitivity(S_rows, specs, out):
    facs = list(STRESS)
    mets = [("V_any", "violación (%)"), ("false_cancel", "cancel. falsa (%)"), ("infeas", "QP infactible (%)")]
    fig, axes = plt.subplots(len(mets), len(facs), figsize=(COL2, 4.6), sharey="row")
    hl = {"B9": BLUE, "B3": ORANGE, "B5": AQUA, "B1": INK2}
    table = {}
    for j, fac in enumerate(facs):
        levels = STRESS[fac]
        for i, (mk, mlab) in enumerate(mets):
            ax = axes[i, j]
            for s in specs:
                ys = []
                for lv in levels:
                    rs = [r for r in S_rows if r["spec"] == s and r["tag"] == f"{fac}={lv}"]
                    v = [r["n_infeas"] > 0 for r in rs] if mk == "infeas" else [r[mk] for r in rs]
                    v = [x for x in v if x == x and x is not None]
                    ys.append(100 * np.mean(v) if v else np.nan)
                    table[(s, fac, lv, mk)] = ys[-1]
                b = base(s)
                c = hl.get(b, "#c9c8c2")
                ax.plot(levels, ys, "-o" if b in hl else "-", color=c, lw=1.3 if b == "B9" else 0.8, ms=2.5,
                        zorder=3 if b in hl else 1, label=NAME[b] if (i == 0 and j == 0 and b in hl) else None)
            if i == 0:
                ax.set_title(STRESS_ES[fac], fontsize=6.3)
            if j == 0:
                ax.set_ylabel(mlab, fontsize=6.5)
            ax.tick_params(labelsize=5.5)
    axes[0, 0].legend(fontsize=5.5, loc="upper left")
    fig.text(0.5, -0.01, "gris: resto de comparadores (B0, B2, B4, B6, B7, B8, A1). Embestida: sólo familia vii.",
             ha="center", fontsize=5.8, color=INK2)
    fig.tight_layout()
    save(fig, out, "g1_sensitivity")
    return table


def fig_feasibility(T, specs, out):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    ys = np.arange(len(specs))[::-1]
    data = {}
    for y, s in zip(ys, specs):
        rs = T[s]
        S = summarize_g1(rs)
        data[s] = S
        col = BLUE if base(s) == "B9" else INK2
        r1 = S["infeas_ep"]
        axes[0].plot([100 * r1["lo"], 100 * r1["hi"]], [y, y], color=col, lw=1); axes[0].plot(100 * r1["rate"], y, "o", color=col, ms=3.5)
        durs = np.array([r["iv_max"] for r in rs if r["n_iv"] > 0]) * 1e3
        if len(durs):
            q = np.percentile(durs, [25, 50, 75, 95])
            axes[1].plot([q[0], q[2]], [y, y], color=col, lw=2.2, solid_capstyle="butt")
            axes[1].plot([q[2], q[3]], [y, y], color=col, lw=0.7)
            axes[1].plot(q[1], y, "|", color=INK, ms=6, mew=1.0)
        r2 = S["iv_viol"]
        if r2["n"]:
            axes[2].plot([100 * r2["lo"], 100 * r2["hi"]], [y, y], color=col, lw=1); axes[2].plot(100 * r2["rate"], y, "o", color=col, ms=3.5)
    for ax, t in zip(axes, ("episodios con pérdida\nde factibilidad (%)", "duración del intervalo más largo\n(ms; caja p25–p75, línea hasta p95)",
                            "violación durante/tras la pérdida\n(% de episodios con pérdida)")):
        ax.set_title(t, fontsize=6.5); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([NAME[base(s)] for s in specs], fontsize=6)
    axes[1].set_yticks([]); axes[2].set_yticks([])
    save(fig, out, "g1_feasibility")
    return data


def fig_controlled(C, out):
    rows, cells, specs = C["rows"], C["cells"], C["specs"]
    show = [s for s in specs if base(s) in ("B2", "B3", "B5", "B9", "A2", "B8")]
    fig, axes = plt.subplots(2, len(show), figsize=(COL2, 3.4), sharey="row")
    import matplotlib.cm as cm
    ps = sorted({c["p"] for c in cells if c["kind"] == "grid"})
    cmap = plt.get_cmap("Blues")
    out_tab = {}
    for j, s in enumerate(show):
        for pi, p in enumerate(ps):
            idx = [i for i, c in enumerate(cells) if c["kind"] == "grid" and c["p"] == p]
            Hs, rel, th = [], [], []
            for i in idx:
                rs = [r for r in rows if r["spec"] == s and r["tag"] == str(i) and r["clamp_applied"]]
                Hs.append(cells[i]["H"])
                rel.append(100 * np.mean([r["clamp_release"] for r in rs]) if rs else np.nan)
                th.append(100 * np.nanmean([r["clamp_theta_d"] for r in rs]))
                out_tab[(s, p, round(cells[i]["a"], 2))] = (Hs[-1], rel[-1], th[-1])
            c = cmap(0.35 + 0.65 * pi / max(len(ps) - 1, 1))
            axes[0, j].plot(Hs, rel, "-o", color=c, ms=2.5, lw=1, label=f"p={p}" if j == 0 else None)
            axes[1, j].plot(Hs, th, "-o", color=c, ms=2.5, lw=1)
        axes[0, j].set_title(NAME[base(s)], fontsize=6.8)
        axes[1, j].set_xlabel("entropía $H$", fontsize=6.5)
        for r_ in (0, 1):
            axes[r_, j].set_xlim(0, 1.05)
    axes[0, 0].set_ylabel("liberación (%)"); axes[1, 0].set_ylabel("$\\bar\\theta_d$ (cm)")
    axes[0, 0].legend(fontsize=5.3, title="$p_W$ fijo", title_fontsize=5.5, loc="upper left")
    fig.text(0.5, -0.02, "Creencia impuesta durante 0,8 s en contacto con un humano que NO se retira: toda liberación es falsa. "
             "Cada curva mantiene $p_W$ constante y aumenta $H$. Sólo episodios en que la creencia llegó a imponerse.",
             ha="center", fontsize=5.8, color=INK2)
    fig.tight_layout()
    save(fig, out, "g1_controlled")
    # iso-entropía: misma H, distinta p
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True)
    iso = {}
    for ax, H0 in zip(axes, ("iso0.4", "iso0.6")):
        idx = [i for i, c in enumerate(cells) if c["kind"] == H0]
        for s in show:
            xs, ys = [], []
            for i in idx:
                rs = [r for r in rows if r["spec"] == s and r["tag"] == str(i) and r["clamp_applied"]]
                xs.append(cells[i]["p"]); ys.append(100 * np.mean([r["clamp_release"] for r in rs]) if rs else np.nan)
                iso[(s, H0, cells[i]["p"])] = ys[-1]
            b = base(s)
            ax.plot(xs, ys, "-o", ms=2.5, color={"B9": BLUE, "B3": ORANGE, "B5": AQUA}.get(b, MUTED),
                    lw=1.3 if b == "B9" else 0.9, label=NAME[b])
        ax.set_title(f"entropía fija $H\\approx${H0[3:]}", fontsize=7); ax.set_xlabel("$p_W$ impuesto")
    axes[0].set_ylabel("liberación (%)"); axes[0].legend(fontsize=5.5, ncol=2)
    save(fig, out, "g1_isoentropy")
    return out_tab, iso


def discrepancy(T, specs, extras):
    """Análisis natural: cada tick (A/C, antes del cierre) se clasifica por H alto/bajo y b_W alto/bajo; se mide
    con qué frecuencia cada método ordena liberar y si la retirada era real (z=W en ese tick)."""
    res = {}
    for s in specs:
        P_ = np.concatenate([e["pred"] for e in extras[s] if len(e["pred"])])
        P_ = P_[P_[:, 5] < 2]                     # modos A y C
        H_, bW, sem, z = P_[:, 1], P_[:, 2], P_[:, 4] > 0.5, P_[:, 6]
        for qn, m in (("H alta, b_W baja", (H_ >= 0.5) & (bW < 0.3)), ("H alta, b_W media", (H_ >= 0.5) & (bW >= 0.3) & (bW < 0.7)),
                      ("H baja, b_W alta", (H_ < 0.5) & (bW >= 0.7)), ("H baja, b_W baja", (H_ < 0.5) & (bW < 0.3))):
            n = int(m.sum())
            res[(s, qn)] = dict(n=n, frac_W=float((z[m] == 2).mean()) if n else np.nan,
                                sem_rate=float(sem[m].mean()) if n else np.nan,
                                fp=float((sem[m] & (z[m] != 2)).sum() / max((z[m] != 2).sum(), 1)) if n else np.nan,
                                tp=float((sem[m] & (z[m] == 2)).sum() / max((z[m] == 2).sum(), 1)) if n else np.nan)
    return res


def fig_discrepancy(D, specs, out):
    qs = ["H alta, b_W baja", "H alta, b_W media", "H baja, b_W alta", "H baja, b_W baja"]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.5), sharey=True)
    ys = np.arange(len(specs))[::-1]
    for ax, q in zip(axes, qs):
        for y, s in zip(ys, specs):
            d = D[(s, q)]
            col = BLUE if base(s) == "B9" else INK2
            ax.barh(y, 100 * d["fp"], color=col, height=0.36, alpha=0.9)
            ax.barh(y - 0.38, 100 * d["tp"], color=ORANGE, height=0.36, alpha=0.9)
        ax.set_xlim(0, 100)
        fw = np.nanmean([D[(s, q)]["frac_W"] for s in specs])
        ax.set_title(f"{q}\n(W real en {100 * fw:.0f} % de los ticks)", fontsize=6.3)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([NAME[base(s)] for s in specs], fontsize=6)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=INK2, label="libera sin retirada real (% de ticks no-W; B9 en azul)"),
                        Patch(color=ORANGE, label="libera con retirada real (% de ticks W)")],
               loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, -0.06), frameon=False)
    save(fig, out, "g1_discrepancy")


def fig_latency_calib(T, extras, sp, out):
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for name, c in (("B1", INK2), ("B3", ORANGE), ("B4", AQUA), ("B9", BLUE)):
        ts = np.sort(np.concatenate([e["t_safety"] for e in extras[sp[name]]]))
        ax.plot(ts, np.arange(1, len(ts) + 1) / len(ts), color=c, label=f"{NAME[name]} (p99 {np.percentile(ts, 99):.2f} ms)")
    ax.axvline(P.LATENCY_BUDGET_MS, color=INK, lw=0.7)
    ax.set_xscale("log"); ax.set_xlim(0.2, 300)
    ax.set_xlabel("cómputo por ciclo de la capa de seguridad (ms)"); ax.set_ylabel("fracción de ciclos")
    ax.legend(fontsize=5.8, loc="lower right")
    save(fig, out, "g1_latency")
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.plot([0, 1], [0, 1], color=EDGE, lw=0.8)
    cal = {}
    for lab_, name, col, ls in (("$b_W$ crudo (A4)", "A4", MUTED, (0, (1, 1))), ("$p_W^+$ bins (A8)", "A8", MUTED, "-"),
                                ("$p_W^+$ isotónica (B9)", "B9", BLUE, "-"), ("HSMM $p_W^+$ (A6)", "A6", AQUA, "-")):
        Pm = np.concatenate([e["pred"] for e in extras[sp[name]] if len(e["pred"])])
        p, y = Pm[:, 3], (Pm[:, 6] == 2).astype(int)
        xs, ys, ns = reliability(p, y)
        ax.plot(xs, ys, marker="o", color=col, ls=ls, ms=2.5, label=lab_)
        cal[lab_] = prediction_metrics(p, y)
    ax.set_xlabel("probabilidad predicha"); ax.set_ylabel("frecuencia observada de W")
    ax.legend(fontsize=5.8, loc="upper left")
    save(fig, out, "g1_reliability")
    return cal


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/g1")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    style()
    P_ = lambda n: os.path.join(a.out, n)
    with open(P_("validate.pkl"), "rb") as f:
        V = pickle.load(f)
    with open(P_("test.pkl"), "rb") as f:
        TT = pickle.load(f)
    from .train_g1 import load
    models = load(P_("models.pkl"))
    rows, extras = {}, {}
    for r, e in TT["out"]:
        rows.setdefault(r["spec"], []).append(r)
        extras.setdefault(r["spec"], []).append(e)
    specs, extra = TT["specs"], TT["extra"]
    sp = {base(s): s for s in specs}
    tau9 = sp["B9"].split("@")[1]
    sp["B6ref"] = f"B6@{tau9}@k4"
    outd = os.path.join(a.out, "paper")
    tabd = os.path.join(outd, "tables")
    os.makedirs(tabd, exist_ok=True)
    main_order = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
    abl_order = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
    S = {s: summarize_g1(rows[s], extras[s]) for s in rows}
    nums = dict(chosen=V["chosen"], s_star=V["s_star"], kappa_val=V["kappa"], w_cart=models.w_design,
                hsmm_k=models.hsmm["k"], n_per_method=len(rows[sp["B9"]]))

    # ---- Tabla principal
    trows = []
    for b in main_order + abl_order:
        s = S[sp[b]]
        trows.append([NAME[b] + (f" ($\\tau$={sp[b].split('@')[1].replace('.', '{,}')})" if "@" in sp[b] and not sp[b].split('@')[1].startswith('a') else ""),
                      fpct(s["V_any"]), fpct(s["V_force"], False), fpct(s["V_dist_link"], False), fpct(s["release_fail"]),
                      fpct(s["sens"], False), fpct(s["false_cancel"]), fpct(s["success_auth"], False),
                      fpct(s["infeas_ep"], False), fnum(s["jerk"], 0), fnum(s["J_int"], 3)])
        if b == "B9":
            trows.append("MID")
    tex_table(os.path.join(tabd, "g1_main.tex"),
              "Resultados en el G1 simulado (MuJoCo), test con participantes no vistos ("
              f"{nums['n_per_method']} episodios por método, 9 familias). IC 95\\,\\% Clopper--Pearson.",
              "tab:g1main", ["Método", "Viol. \\%", "Fuerza \\%", "Eslabón \\%", "Lib. fallida \\%", "Sens. \\%",
                             "Canc. falsa \\%", "Éxito \\%", "Infact. \\%", "Jerk p95", "$J_{int}$"], trows, "l" + "r" * 10)

    # ---- Por familia (todas las métricas, todos los métodos)
    tabs = fig_heatmaps(rows, [sp[b] for b in main_order + ["A1", "A9", "A10"]], outd)
    frows = []
    for f in FAMILIES_G1:
        for b in main_order:
            rs = [r for r in rows[sp[b]] if r["family"] == f]
            frows.append([FAM_ES[f] if b == "B0" else "", NAME[b], fpct(rate([r["V_any"] for r in rs]), False),
                          fnum(np.mean([r["peak_F"] for r in rs])), fpct(rate([r["false_cancel"] for r in rs]), False),
                          fpct(rate([r["release_fail"] for r in rs if r["release_applicable"]]), False),
                          fpct(rate([r["success"] for r in rs]), False), fpct(rate([r["n_infeas"] > 0 for r in rs]), False)])
        frows.append("MID")
    tex_table(os.path.join(tabd, "g1_families.tex"), "Resultados por escenario (G1, test).", "tab:g1fam",
              ["Familia", "Método", "Viol. \\%", "$F$ pico (N)", "Canc. falsa \\%", "Lib. fallida \\%", "Éxito \\%",
               "Infact. \\%"], frows[:-1], "llrrrrrr")

    # ---- Contrastes pre-registrados (Holm)
    B9 = rows[sp["B9"]]
    pre = [("H1", "B6", lambda r: float(r["n_infeas"] > 0), "ep. QP infactible", 100, None),
           ("H1", "B6", F_BIN("jerk_p95"), "jerk p95", 1, None),
           ("H1", "B6ref", lambda r: float(r["n_infeas"] > 0), "ep. QP infactible", 100, None),
           ("H1", "B6ref", F_BIN("jerk_p95"), "jerk p95", 1, None),
           ("H1", "A1", lambda r: float(r["n_infeas"] > 0), "ep. QP infactible", 100, None),
           ("H1", "A1", F_BIN("V_any"), "violación física", 100, None),
           ("H2", "B3", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "B3", lambda r: float(detected(r)) if r["withdraw"] else np.nan, "sensibilidad", 100, None),
           ("H2", "B5", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "B4", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "B8", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "A2", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "A11", F_BIN("false_cancel"), "cancelación falsa", 100, None),
           ("H2", "A11", lambda r: float(detected(r)) if r["withdraw"] else np.nan, "sensibilidad", 100, None),
           ("H3", "A5", F_BIN("V_dist_link"), "violación de eslabón", 100, None),
           ("H3", "B1", F_BIN("V_any"), "violación física", 100, None),
           ("H3", "B1", lambda r: float(r["release_fail"]), "liberación fallida", 100, lambda r: r["release_applicable"])]
    cres = [contrast(B9, rows[sp[o]], f, flt) for (_, o, f, _, _, flt) in pre]
    padj = holm([c["p"] for c in cres])
    crow, nums["contrasts"] = [], []
    for (h, o, _, lab_, sc_, _), c, pa in zip(pre, cres, padj):
        oname = NAME[o if o != "B6ref" else "B6"].split()[0] + ("($\\kappa$=4)" if o == "B6ref" else ("*" if o == "B6" else ""))
        crow.append([h, f"B9 vs {oname}", lab_, f"{fnum(sc_ * c['diff'], 2)}", f"[{fnum(sc_ * c['lo'], 2)}, {fnum(sc_ * c['hi'], 2)}]",
                     fp(c["p"]), fp(pa), f"{c['n']}/{c['n_clusters']}"])
        nums["contrasts"].append(dict(h=h, other=o, metric=lab_, scale=sc_, **c, p_holm=float(pa)))
    tex_table(os.path.join(tabd, "g1_contrasts.tex"),
              "Contrastes pre-registrados en el G1 (B9 $-$ X; pp o m/s$^3$). Bootstrap pareado remuestreando "
              "participantes (4000 réplicas); $p_{Holm}$ sobre los 17 contrastes. B6*: mejor ley exógena elegida en validación.",
              "tab:g1contrasts", ["Hip.", "Contraste", "Métrica", "Dif.", "IC 95\\,\\%", "$p$", "$p_{Holm}$", "$n$/part."],
              crow, "lllrrrrr")

    # ---- Atribución
    pts, strat = fig_attribution(rows, outd, sp, extra)
    nums["kappa_test"] = pts
    nums["strata"] = {f"{k[0]}|{k[1]}": v for k, v in strat.items()}
    arows = []
    for s in sorted(set([x for x in extra + [sp["B6"]] if base(x) == "B6"]),
                    key=lambda s: float("inf") if s.endswith("kinf") else float(s.split("@k")[1])) + [sp["A1"], sp["A10"], sp["A9"], sp["B9"]]:
        Sx = S[s]
        c1 = contrast(B9, rows[s], lambda r: float(r["n_infeas"] > 0)) if s != sp["B9"] else None
        c2 = contrast(B9, rows[s], F_BIN("jerk_p95")) if s != sp["B9"] else None
        arows.append([lab_tex(s), fpct(Sx["infeas_ep"], False), fnum(Sx["jerk"], 0), fpct(Sx["V_any"], False),
                      fpct(Sx["false_cancel"], False), fnum(100 * Sx["gov_frac"], 1) if base(s) in ("B9", "A9", "A10") else "--",
                      "--" if c1 is None else f"{fnum(100 * c1['diff'], 1)} [{fnum(100 * c1['lo'], 1)}, {fnum(100 * c1['hi'], 1)}]",
                      "--" if c2 is None else f"{fnum(c2['diff'], 1)} [{fnum(c2['lo'], 1)}, {fnum(c2['hi'], 1)}]"])
    tex_table(os.path.join(tabd, "g1_attribution.tex"),
              "Atribución al mecanismo de viabilidad. Todas las variantes comparten código, señales y umbral; sólo cambia "
              "cómo evoluciona $\\theta$. ``Gob. activo'': \\% de ciclos en que el QP conjunto se aparta de la referencia "
              "$\\kappa(\\theta^\\star-\\theta)$.", "tab:g1attr",
              ["Variante", "Infact. \\%", "Jerk p95", "Viol. \\%", "Canc. falsa \\%", "Gob. activo \\%",
               "B9$-$X infact. (pp)", "B9$-$X jerk"], arows, "lrrrrrrr")

    # ---- Pérdida de factibilidad
    fe = fig_feasibility(rows, [sp[b] for b in main_order[1:] + ["A1", "A9", "A10"]], outd)
    frw = []
    for b in main_order[1:] + ["A1", "A9", "A10"]:
        s = fe[sp[b]]
        ivs = np.array([r["iv_max"] for r in rows[sp[b]] if r["n_iv"] > 0]) * 1e3
        frw.append([NAME[b], fpct(s["infeas_ep"], False), fnum(s["iv_n"], 2),
                    fnum(np.median(ivs), 0) if len(ivs) else "--", fnum(np.percentile(ivs, 95), 0) if len(ivs) else "--",
                    fpct(s["iv_unrec"], False), fpct(s["iv_viol"], False)])
    tex_table(os.path.join(tabd, "g1_feasibility.tex"),
              "Comportamiento ante la pérdida temporal de factibilidad (test). Intervalo: desde el primer ciclo infactible "
              "hasta que el QP vuelve a ser factible con todas las barreras estimadas $\\ge0$.", "tab:g1feas",
              ["Método", "Ep. con pérdida \\%", "Intervalos/ep.", "Duración mediana (ms)", "p95 (ms)",
               "Sin recuperar \\%", "Violación \\%"], frw, "lrrrrrr")

    # ---- Frontera, sensibilidad, controlado, discrepancia, latencia, calibración
    fig_frontier(V, rows, outd, specs)
    if os.path.exists(P_("sensitivity.pkl")):
        with open(P_("sensitivity.pkl"), "rb") as f:
            SS = pickle.load(f)
        stab = fig_sensitivity(SS["rows"], SS["specs"], outd)
        srow = []
        for fac, lvls in STRESS.items():
            for lv in (lvls[0], lvls[-1]):
                srow.append([STRESS_ES[fac], f"{lv:g}"] + [fnum(stab.get((s, fac, lv, "V_any"), np.nan), 0) + "/" +
                                                           fnum(stab.get((s, fac, lv, "false_cancel"), np.nan), 0)
                                                           for s in SS["specs"] if base(s) in ("B1", "B3", "B5", "B6", "B9", "A1")])
        tex_table(os.path.join(tabd, "g1_sensitivity.tex"),
                  "Sensibilidad a errores de percepción y perturbaciones: violación\\,\\% / cancelación falsa\\,\\% en el "
                  "nivel más leve y el más severo de cada factor.", "tab:g1sens",
                  ["Factor", "Nivel"] + [NAME[base(s)].split()[0] for s in SS["specs"] if base(s) in ("B1", "B3", "B5", "B6", "B9", "A1")],
                  srow, "ll" + "r" * 6)
        nums["sensitivity"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in stab.items()}
    if os.path.exists(P_("controlled.pkl")):
        with open(P_("controlled.pkl"), "rb") as f:
            CC = pickle.load(f)
        ctab, iso = fig_controlled(CC, outd)
        nums["controlled"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ctab.items()}
        nums["isoentropy"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in iso.items()}
        crow2 = []
        for s in CC["specs"]:
            if base(s) not in ("B2", "B3", "B5", "B9", "A2", "B8"):
                continue
            for p in (0.10, 0.35):
                vals = [ctab[(s, p, aa)] for aa in (0.0, 0.15, 0.3, 0.5)]
                crow2.append([NAME[base(s)], f"{p:g}"] + [fnum(v[1], 0) for v in vals] +
                             [fnum(vals[-1][2] - vals[0][2], 1)])
        tex_table(os.path.join(tabd, "g1_controlled.tex"),
                  "Experimento controlado: probabilidad de liberación (\\%) con $p_W$ fijo y entropía creciente "
                  "($a$ = fracción de la masa no-W asignada a H). Última columna: cambio del margen medio $\\theta_d$ (cm) "
                  "entre $a=0$ y $a=0{,}5$. El humano no se retira: toda liberación es falsa.", "tab:g1ctrl",
                  ["Método", "$p_W$", "$a$=0", "0,15", "0,30", "0,50", "$\\Delta\\bar\\theta_d$"], crow2, "lrrrrrr", wide=False)
    D = discrepancy(rows, [sp[b] for b in ("B2", "B3", "B4", "B5", "B8", "B9", "A2", "A11")], extras)
    fig_discrepancy(D, [sp[b] for b in ("B2", "B3", "B4", "B5", "B8", "B9", "A2", "A11")], outd)
    nums["discrepancy"] = {f"{k[0]}|{k[1]}": v for k, v in D.items()}
    cal = fig_latency_calib(rows, extras, sp, outd)
    nums["calibration"] = cal
    lrow = [[NAME[b]] + [fnum(x, 3) for x in S[sp[b]]["lat"][:3]] + [fnum(S[sp[b]]["lat"][3], 1),
            fnum(S[sp[b]]["qp"][0], 3), fnum(S[sp[b]]["qp"][1], 3), fnum(100 * S[sp[b]]["over_budget"], 2)]
            for b in main_order + ["A6", "A9", "A10"]]
    tex_table(os.path.join(tabd, "g1_latency.tex"),
              "Coste por ciclo de la capa de seguridad en el G1 (ms; Python, 11 procesos concurrentes). La latencia "
              f"sensor$\\to$actuador modelada añade {1000 * 0.033:.0f}\\,ms de percepción visual y la dinámica del lazo de par. "
              "En total, 3 de 4{,}4 millones de ciclos superaron 150\\,ms (todos de A9; picos aislados en una máquina compartida).",
              "tab:g1lat", ["Método", "p50", "p95", "p99", "máx", "QP p50", "QP p99", "\\% $>$150 ms"], lrow, "lrrrrrrr")
    if os.path.exists(P_("equivalence.pkl")):
        with open(P_("equivalence.pkl"), "rb") as f:
            EQ = pickle.load(f)
        nums["equivalence"] = dict(n=len(EQ), identical=sum(r["first_div"] is None for r in EQ),
                                   max_du_before=max(r["max_du_before"] for r in EQ),
                                   div_infeasible=sum(1 for r in EQ if r["first_div"] is not None))
    nums["summary"] = {s: {k: (v if not isinstance(v, dict) else v.get("rate")) for k, v in S[s].items()} for s in S}
    if not a.no_render:
        bk, recs = fig_scene(rows, models, outd, sp)
        fig_timeseries(models, outd, bk, sp, recs)
        nums["scene_episode"] = bk
    with open(os.path.join(outd, "numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[report] listo:", sorted(os.listdir(os.path.join(outd, "figures"))))


if __name__ == "__main__":
    main()
