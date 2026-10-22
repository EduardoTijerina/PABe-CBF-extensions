"""Figuras, tablas LaTeX y cifras para el manuscrito a partir de results/paper/exp.pkl.

Uso: python -m pabe_cbf.paper_figs --out results/paper
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

from . import params as P
from .metrics import summarize, rate, cluster_paired_bootstrap, holm, prediction_metrics
from .scenarios import Participant, Scenario, FAMILIES
from .sim import Episode, method_from_spec
from .sweep import detected
from .train import load

# ---- estilo (paleta validada: 3 categóricos que pasan CVD en todos los pares; gris = referencia) ----
INK, INK2, MUTED, GRID, EDGE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_METHOD = {"B9": BLUE, "B3": ORANGE, "B5": AQUA, "A8": MUTED, "A1": MUTED, "B1": INK2}
C_Z = {0: BLUE, 1: AQUA, 2: ORANGE}
Z_NAME = {0: "E (interesado)", 1: "H (dubitativo)", 2: "W (retirándose)"}
COL1, COL2 = 3.5, 7.16
FAM_ES = {"i_stable": "(i) estable", "ii_withdraw_pre": "(ii) retirada pre-contacto",
          "iii_withdraw_contact": "(iii) retirada en contacto", "iv_hesitation": "(iv) duda inocua",
          "v_occlusion": "(v) oclusión/táctil tardío", "vi_perturbation": "(vi) perturbación"}
LABEL = {"B0": "B0 nominal", "B1": "B1 CBF fijo", "B2": "B2 HMM-MAP", "B3": "B3 HMM-$H(b)$",
         "B4": "B4 BCBF", "B5": "B5 CRC-CBF", "B6": "B6 TV-CBF", "B7": "B7 SHIELD", "B8": "B8 RCIP",
         "B9": "B9 PABe-CBF", "A1": "A1 sin gobernador", "A2": "A2 $H(b)$ global", "A3": "A3 sin $p_W^+$",
         "A4": "A4 $b_W$ crudo", "A5": "A5 barrera única", "A7": "A7 sin táctil", "A8": "A8 $p_W^+$ por bins"}
LABEL_TEX = {k: v.replace("$H(b)$", "$H(b)$") for k, v in LABEL.items()}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": EDGE,
        "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})


def base(spec):
    return spec.split("@")[0]


def save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".pdf"))
    fig.savefig(os.path.join(d, name + ".png"))
    plt.close(fig)


def fnum(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    v = f"{x:.{nd}f}"
    if v.startswith("-"):
        v = "$-$" + v[1:] if float(v) != 0 else v[1:]
    return v.replace(".", "{,}")


def fp(p):
    return "$<$0{,}001" if p < 0.001 else fnum(p, 3)


def fpct(r, ci=True):
    if r["n"] == 0:
        return "--"
    s = fnum(100 * r["rate"])
    if ci:
        s += f" \\scriptsize[{fnum(100 * r['lo'])}, {fnum(100 * r['hi'])}]"
    return s


# ------------------------------------------------------------------------------------
def key(r):
    return (r["family"], r["pid"], r["seed"])


def method_summary(E, spec):
    rows, ex = E["rows"][spec], E["extras"].get(spec)
    s = summarize(rows, ex) if ex else {}
    s["sens"] = rate([detected(r) for r in rows if r["withdraw"]])
    return s


def contrast(E, a, b, metric, filt=None):
    ia = {key(r): r for r in E["rows"][a]}
    ib = {key(r): r for r in E["rows"][b]}
    ks = sorted(set(ia) & set(ib))
    get = metric if callable(metric) else (lambda r: float(r[metric]) if r[metric] is not None else np.nan)
    va, vb, cl = [], [], []
    for k in ks:
        if filt and not filt(ia[k]):
            continue
        va.append(get(ia[k])); vb.append(get(ib[k])); cl.append(k[1])
    return cluster_paired_bootstrap(va, vb, cl)


# ------------------------------------------------------------------------------------
def pick_episode(E, b9, b1):
    """Episodio representativo de retirada en contacto (familia iii): mayor diferencia de fuerza pico
    B1 - B9 entre los que ambos llegaron a contacto. Se elige por criterio explícito, no a mano."""
    i1 = {key(r): r for r in E["rows"][b1] if r["family"] == "iii_withdraw_contact"}
    best, bk = -1, None
    for r in E["rows"][b9]:
        k = key(r)
        if k in i1 and r["release_applicable"] and i1[k]["release_applicable"] and not r["shift"]:
            d = i1[k]["peak_F"] - r["peak_F"]
            if d > best:
                best, bk = d, k
    return bk


def record(spec, k, models, shift=False):
    fam, pid, seed = k
    sc = Scenario(fam, Participant(pid, shift=shift), seed)
    ep = Episode(method_from_spec(spec), sc, models)
    ep.record = []
    res, _, _ = ep.run()
    return ep.record, sc, res


def draw_scene(ax, rec, title, theta_show=True):
    ax.set_aspect("equal")
    ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.52, 0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # cuerpos
    ax.add_patch(FancyBboxPatch((-0.16, -0.36), 0.12, 0.32, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#efeeea", ec=EDGE, lw=0.6, zorder=1))
    ax.add_patch(Circle(P.TORSO, 0.13, fc="#efeeea", ec=EDGE, lw=0.6, zorder=1))
    ax.text(-0.10, -0.42, "robot", ha="center", fontsize=6, color=MUTED)
    ax.text(P.TORSO[0], -0.2, "persona", ha="center", fontsize=6, color=MUTED)
    hum = rec["hum"]
    hx = [P.H_SHOULDER, hum["helbow"], hum["hwrist"], hum["hand"]]
    ax.plot(*np.array(hx).T, color="#a9a8a2", lw=3.2, solid_capstyle="round", zorder=2)
    ax.plot(*hum["hand"], "o", ms=5.5, color="#a9a8a2", mec="white", mew=0.8, zorder=3)
    pts = rec["pts"]
    rx = [P.SHOULDER, pts["elbow"], pts["wrist"], pts["palm"]]
    ax.plot(*np.array(rx).T, color="#3d3d3a", lw=3.2, solid_capstyle="round", zorder=4)
    for p_ in rx[:3]:
        ax.plot(*p_, "o", ms=3, color="white", mec="#3d3d3a", mew=0.8, zorder=5)
    ax.plot(*pts["palm"], "o", ms=5.5, color=BLUE, mec="white", mew=0.8, zorder=5)
    if rec["mode"] in ("A", "R") and not rec["engaged"]:
        r_ = P.D_SEP_PALM + P.EPS_D + rec["theta"][0]
        ax.add_patch(Circle(hum["hand"], r_, fill=False, ec=BLUE, lw=0.8, ls=(0, (2, 1.5)), zorder=3))
        ax.add_patch(Circle(hum["hand"], P.D_SEP_PALM, fill=False, ec=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=3))
    if rec["engaged"] and abs(rec["F"]) > 0.3:
        n = pts["palm"] - hum["hand"]
        n = n / (np.linalg.norm(n) + 1e-9)
        L = 0.012 * abs(rec["F"])
        ax.annotate("", xy=pts["palm"] - n * L * np.sign(rec["F"]), xytext=pts["palm"],
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, mutation_scale=7), zorder=6)
    ax.set_title(title, fontsize=6.8, color=INK, pad=2)
    ax.text(0.5, -0.02, f"modo {rec['mode']} · |F| = {abs(rec['F']):.1f} N\n$\\theta_d$ = {100 * rec['theta'][0]:.1f} cm",
            transform=ax.transAxes, ha="center", va="top", fontsize=6, color=INK2)


def fig_scene(E, models, out, k, b1, b9):
    rec9, sc, _ = record(b9, k, models)
    rec1, _, _ = record(b1, k, models)
    t_eng = next((r["t"] for r in rec9 if r["engaged"]), None)
    t_C = next((r["t"] for r in rec9 if r["mode"] == "C"), rec9[0]["t"])
    times = [max(0.0, t_C - 0.35), (t_eng or t_C) + 0.4, sc.t_W + 0.25, sc.t_W + 0.55]
    names = ["aproximación", "contacto autorizado", "retirada +0,25 s", "retirada +0,55 s"]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.05))
    for row, (rec, lab) in enumerate(((rec1, LABEL["B1"]), (rec9, LABEL["B9"]))):
        for c, (t, nm) in enumerate(zip(times, names)):
            i = min(int(round(t / P.DT)), len(rec) - 1)
            draw_scene(axes[row, c], rec[i], f"{nm}\nt = {rec[i]['t']:.2f} s" if row == 0 else
                       f"t = {rec[i]['t']:.2f} s")
        axes[row, 0].text(-0.08, 0.5, lab, transform=axes[row, 0].transAxes, rotation=90, va="center",
                          ha="right", fontsize=7.5, color=INK)
    ax = axes[0, 0]
    ax.plot([-0.15, -0.05], [0.44, 0.44], color=INK2, lw=1)
    ax.text(-0.10, 0.455, "10 cm", ha="center", va="bottom", fontsize=5.5, color=INK2)
    fig.subplots_adjust(wspace=0.04, hspace=0.32)
    save(fig, out, "fig_scene")
    return rec9, rec1, sc


def mode_bands(ax, rec, label=False):
    t = np.array([r["t"] for r in rec])
    modes = [r["mode"] for r in rec]
    shade = {"A": None, "C": "#f2f1ec", "R": "#e7eef9"}
    s = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[s]:
            if shade[modes[s]]:
                ax.axvspan(t[s], t[i - 1], color=shade[modes[s]], lw=0, zorder=0)
            if label:
                ax.text((t[s] + t[i - 1]) / 2, 1.02, modes[s], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=6.5, color=INK2)
            s = i


def fig_timeseries(E, models, out, k, specs):
    recs = {s: record(s, k, models)[0] for s in specs}
    sc = Scenario(k[0], Participant(k[1]), k[2])
    b9 = [s for s in specs if base(s) == "B9"][0]
    r9 = recs[b9]
    t9 = np.array([r["t"] for r in r9])
    fig, axes = plt.subplots(5, 1, figsize=(COL2, 6.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1.1, 1]))
    for ax in axes:
        mode_bands(ax, r9, label=(ax is axes[0]))
        ax.axvline(sc.t_W, color=ORANGE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axes[0].text(sc.t_W, 0.5, " inicio real de retirada", color=ORANGE, fontsize=6, va="center",
                 transform=axes[0].get_xaxis_transform())
    # (a) creencia
    B = np.array([r["b"] for r in r9])
    for z in range(3):
        axes[0].plot(t9, B[:, z], color=C_Z[z], label=Z_NAME[z])
    axes[0].set_ylabel("creencia $b_t$")
    axes[0].legend(ncol=3, loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=6.5)
    axes[0].set_ylim(-0.03, 1.03)
    # (b) riesgo vs incertidumbre
    tau9 = float(b9.split("@")[1])
    axes[1].plot(t9, [r["p"] for r in r9], color=BLUE, label="$p_W^+$ (B9)")
    axes[1].plot(t9, B[:, 2], color=BLUE, lw=0.8, ls=(0, (1, 1)), label="$b_W$ crudo")
    axes[1].plot(t9, [r["H"] for r in r9], color=MUTED, lw=1.0, label="$H_t$")
    axes[1].plot(t9, [r["Eexc"] for r in r9], color=AQUA, label="$E^{exc}_t$")
    axes[1].axhline(tau9, color=BLUE, lw=0.6, ls=(0, (4, 2)))
    axes[1].text(t9[-1], tau9, f"τ={tau9:g}", fontsize=6, color=BLUE, ha="right", va="bottom")
    axes[1].set_ylabel("señal")
    axes[1].legend(ncol=4, loc="center left", bbox_to_anchor=(0.02, 0.55), fontsize=6.5)
    axes[1].set_ylim(-0.03, 1.08)
    # (c) margen theta_d
    a1 = [s for s in specs if base(s) == "A1"]
    axes[2].plot(t9, [100 * r["theta_star"][0] for r in r9], color=MUTED, lw=0.9, ls=(0, (2, 1.5)),
                 label="objetivo $\\theta_d^\\star$")
    axes[2].plot(t9, [100 * r["theta"][0] for r in r9], color=BLUE, label="B9: $\\theta_d$ en el QP")
    if a1:
        ra = recs[a1[0]]
        axes[2].plot([r["t"] for r in ra], [100 * r["theta"][0] for r in ra], color=INK2, lw=0.9,
                     label="A1: $\\theta_d\\leftarrow\\theta_d^\\star$")
    axes[2].set_ylabel("$\\theta_d$ (cm)")
    axes[2].legend(ncol=3, loc="upper left", fontsize=6.5)
    # (d) fuerza
    for s in specs:
        if base(s) in ("A1",):
            continue
        rr = recs[s]
        axes[3].plot([r["t"] for r in rr], [abs(r["F"]) for r in rr], color=C_METHOD.get(base(s), MUTED),
                     label=LABEL[base(s)], lw=1.4 if base(s) == "B9" else 1.0)
    axes[3].axhline(P.F_PHYS, color=INK, lw=0.7)
    axes[3].text(t9[0], P.F_PHYS, " $F_{phys}$", fontsize=6, va="bottom", color=INK)
    axes[3].axhline(P.F_SOC0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    axes[3].text(t9[0], P.F_SOC0, " $F_{soc}$ nominal", fontsize=6, va="bottom", color=MUTED)
    axes[3].set_ylabel("|F| (N)")
    axes[3].legend(ncol=3, loc="upper left", bbox_to_anchor=(0.18, 1.0), fontsize=6.5)
    # (e) aceleración comandada del efector (corrección de la acción)
    for s in specs:
        rr = recs[s]
        u = np.array([r["u"] for r in rr]); un = np.array([r["u_nom"] for r in rr])
        axes[4].plot([r["t"] for r in rr], np.linalg.norm(u - un, axis=1), color=C_METHOD.get(base(s), MUTED),
                     lw=1.3 if base(s) == "B9" else 0.9, label=LABEL[base(s)],
                     ls="-" if base(s) != "A1" else (0, (2, 1)))
    axes[4].set_ylabel("$\\|u-u_{nom}\\|$")
    axes[4].set_xlabel("tiempo (s)")
    axes[4].legend(ncol=4, loc="upper left", fontsize=6.5)
    for i, ax in enumerate(axes):
        ax.text(-0.095, 1.0, f"({'abcde'[i]})", transform=ax.transAxes, fontsize=8, color=INK, va="top")
    axes[-1].set_xlim(t9[0], t9[-1])
    fig.align_ylabels(axes)
    save(fig, out, "fig_timeseries")


def fig_frontier(E, out):
    fv, ft, ch, s_star = E["front_val"], E["front_test"], E["chosen"], E["s_star"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), sharey=True)
    for ax, fr, ttl in ((axes[0], fv, "validación (selección de τ)"), (axes[1], ft, "test v2 (participantes no vistos)")):
        for b, col, ls in (("B3", ORANGE, "-"), ("B5", AQUA, "-"), ("A8", MUTED, (0, (3, 2))), ("B9", BLUE, "-")):
            pts = [p for p in fr[b] if p["sens"]["n"]]
            xs = np.array([100 * p["sens"]["rate"] for p in pts]); ys = np.array([100 * p["fc"]["rate"] for p in pts])
            ax.plot(xs, ys, ls=ls, color=col, marker="o", ms=3.5, mec="white", mew=0.6, zorder=3,
                    label=LABEL[b] + (" (v1)" if b == "A8" else ""))
            for p, x, y in zip(pts, xs, ys):
                if abs(p["tau"] - ch[b]) < 1e-9:
                    ax.errorbar([x], [y], xerr=[[x - 100 * p["sens"]["lo"]], [100 * p["sens"]["hi"] - x]],
                                yerr=[[y - 100 * p["fc"]["lo"]], [100 * p["fc"]["hi"] - y]], color=col,
                                lw=0.7, capsize=1.5, zorder=4)
                    ax.plot([x], [y], "o", ms=8, mfc="none", mec=col, mew=1.2, zorder=5)
        ax.axvline(100 * s_star, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        ax.set_title(ttl)
        ax.set_xlabel("sensibilidad de retirada (%)")
    axes[0].set_ylabel("cancelación falsa (% apretones normales)")
    axes[0].legend(loc="upper left", fontsize=6.5)
    axes[1].text(0.02, 0.97, "abajo-derecha = mejor\n○ punto elegido en validación (IC 95 %)",
                 transform=axes[1].transAxes, fontsize=6, color=INK2, va="top")
    save(fig, out, "fig_frontier")


def fig_dots(E, S, specs, out):
    mets = [("V_any", "violación\nfísica (%)", False), ("false_cancel", "cancelación\nfalsa (%)", False),
            ("release_fail", "liberación\nfallida (%)", False), ("infeas_episodes", "episodios con\nQP infactible (%)", False),
            ("success_auth", "éxito\nautorizado (%)", True)]
    fig, axes = plt.subplots(1, len(mets), figsize=(COL2, 2.9), sharey=True)
    ys = np.arange(len(specs))[::-1]
    for ax, (m, lab, up) in zip(axes, mets):
        for y, s in zip(ys, specs):
            r = S[s][m]
            if r["n"] == 0 or r["rate"] != r["rate"]:
                continue
            col = BLUE if base(s) == "B9" else (INK2 if base(s) in ("B1",) else MUTED)
            ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=col, lw=1.0, solid_capstyle="round")
            ax.plot(100 * r["rate"], y, "o", ms=4.2 if base(s) == "B9" else 3.4, color=col, mec="white", mew=0.5)
        ax.set_title(lab, fontsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=-2)
        ax.text(0.98, -0.1, "↑ mejor" if up else "↓ mejor", transform=ax.transAxes, ha="right", fontsize=6, color=MUTED)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABEL[base(s)] for s in specs])
    for t in axes[0].get_yticklabels():
        if t.get_text().startswith("B9"):
            t.set_color(BLUE); t.set_fontweight("bold")
    save(fig, out, "fig_methods")


def fig_ablation(E, b9, abl, out):
    mets = [("false_cancel", "cancelación\nfalsa (pp)", 100, None),
            ("release_fail", "liberación\nfallida (pp)", 100, lambda r: r["release_applicable"]),
            (lambda r: float(r["n_infeas"] > 0), "episodios con\nQP infactible (pp)", 100, None),
            ("jerk_p95", "jerk p95 de\nla palma (m/s³)", 1, None),
            ("V_dist_link", "violación de\neslabón (pp)", 100, None)]
    fig, axes = plt.subplots(1, len(mets), figsize=(COL2, 2.3), sharey=True)
    ys = np.arange(len(abl))[::-1]
    res = {}
    for ax, (m, lab, sc_, filt) in zip(axes, mets):
        for y, s in zip(ys, abl):
            c = contrast(E, s, b9, m, filt)
            res[(base(s), lab)] = c
            sig = c["lo"] > 0 or c["hi"] < 0
            col = INK2 if sig else MUTED
            ax.plot([sc_ * c["lo"], sc_ * c["hi"]], [y, y], color=col, lw=1.0)
            ax.plot(sc_ * c["diff"], y, "o", ms=3.6, color=col, mfc=col if sig else "white", mew=0.8)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(lab, fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABEL[base(s)] for s in abl])
    fig.text(0.5, -0.04, "efecto de cada modificación respecto de B9 (X − B9), media e IC 95 % bootstrap por participante; "
             "relleno = IC excluye 0; > 0 indica empeoramiento", ha="center", fontsize=6.3, color=INK2)
    save(fig, out, "fig_ablation")
    return res


def fig_latency(E, specs, out):
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for s, col, ls in specs:
        ts = np.sort(np.concatenate([e["t_safety"] for e in E["extras"][s]]))
        y = np.arange(1, len(ts) + 1) / len(ts)
        ax.plot(ts, y, color=col, ls=ls, label=f"{LABEL[base(s)]} (p99 {np.percentile(ts, 99):.2f} ms)")
    ax.axvline(P.LATENCY_BUDGET_MS, color=INK, lw=0.7)
    ax.text(P.LATENCY_BUDGET_MS, 0.5, "presupuesto\n150 ms ", ha="right", fontsize=6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 300)
    ax.set_xlabel("tiempo de cómputo por ciclo de la capa de seguridad (ms, escala log)")
    ax.set_ylabel("fracción acumulada de ciclos")
    ax.legend(loc="lower right", fontsize=6)
    save(fig, out, "fig_latency")


def reliability(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    xs, ys, ns = [], [], []
    for k in range(bins):
        m = idx == k
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def fig_reliability(E, sources, out):
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    ax.plot([0, 1], [0, 1], color=EDGE, lw=0.8)
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#f6f5f1", lw=0, zorder=0)
    ax.text(0.97, 0.06, "zona conservadora\n(probabilidad predicha ≥ frecuencia)", ha="right", fontsize=6, color=MUTED)
    out_m = {}
    for lab, s, col, ls in sources:
        p = np.concatenate([e["pred_p"] for e in E["extras"][s]])
        y = np.concatenate([e["pred_y"] for e in E["extras"][s]])
        xs, ys, ns = reliability(p, y)
        ax.plot(xs, ys, color=col, ls=ls, marker="o", ms=3, mec="white", mew=0.5, label=lab)
        out_m[lab] = prediction_metrics(p, y)
        out_m[lab]["under_frac"] = float(np.average(ys > xs + 0.02, weights=ns)) if len(xs) else np.nan
    ax.set_xlabel("probabilidad de retirada predicha")
    ax.set_ylabel("frecuencia observada de W")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=6.3)
    save(fig, out, "fig_reliability")
    return out_m


def gif(rec1, rec9, out):
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = min(len(rec1), len(rec9))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    def upd(i):
        for ax, rec, lab in ((axes[0], rec1, LABEL["B1"]), (axes[1], rec9, LABEL["B9"])):
            ax.clear()
            draw_scene(ax, rec[i], f"{lab} · t = {rec[i]['t']:.2f} s")
    an = FuncAnimation(fig, upd, frames=range(0, n, 3), interval=50)
    d = os.path.join(out, "figures")
    an.save(os.path.join(d, "sim_B1_vs_B9.gif"), writer=PillowWriter(fps=20), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------
def tex_table(path, caption, label, header, rows, colspec, notes=None, wide=True):
    env = "table*" if wide else "table"
    L = [f"\\begin{{{env}}}[!t]", "\\centering\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{3.2pt}", "\\resizebox{\\linewidth}{!}{%",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        L.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}"]
    if notes:
        L.append(f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    L.append(f"\\end{{{env}}}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    style()
    with open(os.path.join(a.out, "exp.pkl"), "rb") as f:
        E = pickle.load(f)
    models = load(os.path.join(a.out, "models.pkl"))
    tabd = os.path.join(a.out, "tables")
    os.makedirs(tabd, exist_ok=True)
    ms = E["main_specs"]
    spec = {base(s): s for s in ms}
    order = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
    abl = ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "B6"]
    S = {s: method_summary(E, s) for s in ms}
    nums = dict(chosen=E["chosen"], s_star=E["s_star"], args=E["args"],
                n_episodes_main=len(E["rows"][spec["B9"]]))

    # ---------- Tabla 1: resultados principales ----------
    rows = []
    for b in order:
        s = S[spec[b]]
        tau = f" ($\\tau$={spec[b].split('@')[1].replace('.', '{,}')})" if "@" in spec[b] else ""
        rows.append([LABEL_TEX[b] + tau, fpct(s["V_any"]), fpct(s["V_unauth_contact"], False), fnum(s["peak_F_p95"]),
                     fpct(s["release_fail"]), fpct(s["sens"], False), fpct(s["false_cancel"]),
                     fpct(s["success_auth"], False), fpct(s["infeas_episodes"], False), fnum(s["jerk_p95_mean"], 0),
                     fnum(s["J_int_mean"], 3)])
    tex_table(os.path.join(tabd, "tab_main.tex"),
              "Resultados en el test v2 (participantes no vistos). Porcentajes por episodio con IC 95\\,\\% de "
              "Clopper--Pearson. $F_{p95}$: percentil 95 de la fuerza pico por episodio. Sens.: retiradas detectadas "
              "por la regla semántica en $\\le$1\\,s. Infact.: episodios con $\\ge$1 ciclo QP infactible.",
              "tab:main",
              ["Método", "Viol. física \\%", "Cont. no aut. \\%", "$F_{p95}$ (N)", "Lib. fallida \\%", "Sens. \\%",
               "Canc. falsa \\%", "Éxito \\%", "Infact. \\%", "Jerk p95", "$J_{int}$"],
              rows, "l" + "r" * 10,
              notes="B0 sólo en simulación. B2, B4, B7 y B8 usan su regla nativa sin barrido de umbral; "
                    "B3, B5 y B9 usan el umbral elegido en validación para igualar sensibilidad. "
                    "Jerk en m/s$^3$; $J_{int}$ adimensional (normalizado por $u_{max}$).")

    # ---------- Tabla 2: H2 a igual sensibilidad ----------
    h2 = []
    for b in ("B3", "B5", "A8", "B9"):
        s = S[spec[b]]
        h2.append([LABEL_TEX[b], spec[b].split("@")[1].replace(".", "{,}"), fpct(s["sens"]), fpct(s["false_cancel"]),
                   fpct(s["release_fail"]), fpct(s["success_auth"])])
    tex_table(os.path.join(tabd, "tab_h2.tex"),
              f"Puntos de operación elegidos en validación para sensibilidad $\\ge$ {fnum(100 * E['s_star'])}\\,\\% "
              "con mínima cancelación falsa, evaluados en test v2.", "tab:h2",
              ["Método", "$\\tau$", "Sensibilidad \\%", "Cancelación falsa \\%", "Liberación fallida \\%", "Éxito \\%"],
              h2, "lrrrrr", wide=False)

    # ---------- Tabla 3: contrastes pre-registrados con Holm ----------
    pre = [("H1", spec["B9"], spec["B6"], lambda r: float(r["n_infeas"] > 0), "episodios con QP infactible", 100, None),
           ("H1", spec["B9"], spec["B6"], "jerk_p95", "jerk p95 (m/s$^3$)", 1, None),
           ("H1", spec["B9"], spec["B6"], "J_int", "$J_{int}$", 1, None),
           ("H2", spec["B9"], spec["B3"], "false_cancel", "cancelación falsa", 100, None),
           ("H2", spec["B9"], spec["B5"], "false_cancel", "cancelación falsa", 100, None),
           ("H2", spec["B9"], spec["B3"], lambda r: float(detected(r)) if r["withdraw"] else np.nan, "sensibilidad", 100, None),
           ("H3", spec["B9"], spec["B1"], "V_dist_link", "violación de eslabón", 100, None),
           ("H3", spec["B9"], spec["B1"], "V_unauth_contact", "contacto no autorizado", 100, None),
           ("H3", spec["B9"], spec["B1"], "V_any", "violación física (cualquiera)", 100, None),
           ("H4", spec["B9"], spec["B1"], lambda r: np.nan, "", 1, None)]
    pre = [p for p in pre if p[0] != "H4"]
    cres = [contrast(E, a_, b_, m, f) for (_, a_, b_, m, _, _, f) in pre]
    padj = holm([c["p"] for c in cres])
    rows3, nums["contrasts"] = [], []
    for (h, a_, b_, m, lab, sc_, _), c, pa in zip(pre, cres, padj):
        u = " pp" if sc_ == 100 else ""
        nd = 3 if "J_{int}" in lab else 2
        rows3.append([h, f"{LABEL_TEX[base(a_)].split()[0]} vs {LABEL_TEX[base(b_)].split()[0]}", lab,
                      f"{fnum(sc_ * c['diff'], nd)}{u}", f"[{fnum(sc_ * c['lo'], nd)}, {fnum(sc_ * c['hi'], nd)}]",
                      fp(c["p"]), fp(pa), f"{c['n']}/{c['n_clusters']}"])
        nums["contrasts"].append(dict(h=h, a=a_, b=b_, metric=lab, scale=sc_, **c, p_holm=float(pa)))
    tex_table(os.path.join(tabd, "tab_contrasts.tex"),
              "Contrastes pre-registrados (diferencia B9 $-$ X). IC 95\\,\\% y $p$ bilateral por bootstrap pareado "
              "remuestreando participantes (4000 réplicas); $p_{Holm}$ corrige por multiplicidad sobre la familia completa.",
              "tab:contrasts", ["Hip.", "Contraste", "Métrica", "Diferencia", "IC 95\\,\\%", "$p$", "$p_{Holm}$",
                                "$n$ ep./part."], rows3, "lllrrrrr")

    # ---------- Tabla 4: ablaciones ----------
    rows4 = []
    abl_specs = [spec[b] for b in abl]
    for s in abl_specs:
        cs = [contrast(E, s, spec["B9"], "false_cancel"),
              contrast(E, s, spec["B9"], "release_fail", lambda r: r["release_applicable"]),
              contrast(E, s, spec["B9"], lambda r: float(r["n_infeas"] > 0)),
              contrast(E, s, spec["B9"], "jerk_p95"),
              contrast(E, s, spec["B9"], "V_dist_link")]
        sc_ = [100, 100, 100, 1, 100]
        cells = []
        for c, k_ in zip(cs, sc_):
            sig = (c["lo"] > 0 or c["hi"] < 0)
            v = f"{fnum(k_ * c['diff'], 1)} \\scriptsize[{fnum(k_ * c['lo'], 1)}, {fnum(k_ * c['hi'], 1)}]"
            cells.append(("\\textbf{" + v + "}") if sig else v)
        rows4.append([LABEL_TEX[base(s)]] + cells)
    tex_table(os.path.join(tabd, "tab_ablation.tex"),
              "Ablaciones: efecto X $-$ B9 (pp o m/s$^3$) con IC 95\\,\\% bootstrap por participante; en negrita "
              "si el IC excluye 0. Valores positivos indican que eliminar el componente empeora la métrica.",
              "tab:ablation", ["Variante", "Canc. falsa", "Lib. fallida", "Ep. infactibles", "Jerk p95", "Viol. eslabón"],
              rows4, "lrrrrr")

    # ---------- Tabla 5: por familia ----------
    rows5 = []
    for fam in FAMILIES:
        for b in ("B1", "B3", "B5", "B9"):
            rs = [r for r in E["rows"][spec[b]] if r["family"] == fam]
            rows5.append([FAM_ES[fam] if b == "B1" else "", LABEL_TEX[b],
                          fpct(rate([r["V_any"] for r in rs]), False), f"{fnum(np.mean([r['peak_F'] for r in rs]))}",
                          fpct(rate([r["false_cancel"] for r in rs]), False),
                          fpct(rate([r["release_fail"] for r in rs if r["release_applicable"]]), False),
                          fpct(rate([r["success"] for r in rs]), False)])
        rows5.append("MID")
    rows5 = rows5[:-1]
    tex_table(os.path.join(tabd, "tab_families.tex"),
              "Desglose por familia de escenarios (test v2). Éxito: apretón completado si está autorizado y no hay "
              "retirada; ausencia de contacto si no está autorizado.", "tab:families",
              ["Familia", "Método", "Viol. \\%", "$F$ pico medio (N)", "Canc. falsa \\%", "Lib. fallida \\%", "Éxito \\%"],
              rows5, "llrrrrr")

    # ---------- Tabla 6: latencia ----------
    rows6 = []
    for b in order:
        s = S[spec[b]]
        rows6.append([LABEL_TEX[b], fnum(s["lat_p50"], 3), fnum(s["lat_p95"], 3), fnum(s["lat_p99"], 3),
                      fnum(s["lat_max"], 2), fnum(s["qp_p50"], 3) if b != "B0" else "--",
                      fnum(s["qp_p99"], 3) if b != "B0" else "--", fnum(s["hmm_p50"], 3),
                      fnum(100 * s["lat_over_budget"], 2)])
    tex_table(os.path.join(tabd, "tab_latency.tex"),
              "Coste computacional por ciclo de la capa de seguridad (Python 3.12 + quadprog, un núcleo por proceso). "
              "No incluye adquisición, transporte ni actuación.", "tab:latency",
              ["Método", "p50", "p95", "p99", "máx", "QP p50", "QP p99", "HMM p50", "\\% $>$150 ms"], rows6,
              "l" + "r" * 8, notes="Tiempos en ms.")

    # ---------- figuras ----------
    k = pick_episode(E, spec["B9"], spec["B1"])
    nums["episode"] = k
    rec9, rec1, _ = fig_scene(E, models, a.out, k, spec["B1"], spec["B9"])
    fig_timeseries(E, models, a.out, k, [spec["B1"], spec["B3"], spec["A1"], spec["B9"]])
    fig_frontier(E, a.out)
    fig_dots(E, S, [spec[b] for b in order], a.out)
    nums["ablation_fig"] = {f"{k_[0]}|{k_[1]}": v for k_, v in fig_ablation(E, spec["B9"], abl_specs, a.out).items()}
    fig_latency(E, [(spec["B1"], INK2, "-"), (spec["B3"], ORANGE, "-"), (spec["B4"], AQUA, "-"),
                    (spec["B9"], BLUE, "-")], a.out)
    rel = fig_reliability(E, [("$b_W$ crudo (A4)", spec["A4"], MUTED, (0, (1, 1))),
                              ("$p_W^+$ por bins, v1 (A8)", spec["A8"], MUTED, "-"),
                              ("$p_W^+$ isotónica, v2 (B9)", spec["B9"], BLUE, "-"),
                              ("media calibrada (B5)", spec["B5"], AQUA, "-")], a.out)
    nums["calibration"] = rel
    rows7 = [[lab, fnum(v["brier"], 3), fnum(v["nll"], 3), fnum(v["ece"], 3), fnum(v["auroc"], 3),
              fnum(v["recall_fpr5"], 3)] for lab, v in rel.items()]
    tex_table(os.path.join(tabd, "tab_calibration.tex"),
              "Calidad probabilística de la señal de retirada en test v2 (por tick; los ticks están autocorrelados, "
              "por lo que estas cifras son descriptivas). $p_W^+$ pretende ser una cota superior: un buen "
              "resultado es ECE bajo con sesgo conservador (ver Fig.~\\ref{fig:reliability}).", "tab:calibration",
              ["Señal", "Brier", "log-loss", "ECE", "AUROC", "Recall@FPR5\\%"], rows7, "lrrrrr", wide=False)
    if not a.no_gif:
        gif(rec1, rec9, a.out)

    # cifras para el texto
    nums["summary"] = {b: {kk: (vv if not isinstance(vv, dict) else vv.get("rate", vv))
                           for kk, vv in S[spec[b]].items()} for b in order + abl if b in spec}
    with open(os.path.join(a.out, "numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[figs] listo:", sorted(os.listdir(os.path.join(a.out, "figures"))), sorted(os.listdir(tabd)))


if __name__ == "__main__":
    main()
