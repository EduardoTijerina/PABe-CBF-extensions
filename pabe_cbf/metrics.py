"""Métricas de la Tabla 3 y análisis estadístico de la Sec. 7.5 (bootstrap pareado, CP, cota 0-eventos)."""
import numpy as np
from scipy.stats import beta


def clopper_pearson(k, n, conf=0.95):
    if n == 0:
        return (np.nan, np.nan)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


def zero_event_upper(n, conf=0.95):
    """Ec. 16: 0 eventos en n episodios iid -> cota superior unilateral exacta."""
    return 1 - (1 - conf) ** (1 / n)


def rate(x):
    x = np.asarray([v for v in x if v == v], dtype=float)   # descarta NaN
    k, n = int(x.sum()), len(x)
    lo, hi = clopper_pearson(k, n)
    return dict(k=k, n=n, rate=k / n if n else np.nan, lo=lo, hi=hi,
                upper0=zero_event_upper(n) if (n and k == 0) else np.nan)


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def auroc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def recall_at_fpr(p, y, fpr=0.05):
    p, y = np.asarray(p), np.asarray(y)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return np.nan
    thr = np.quantile(p[y == 0], 1 - fpr)
    return float((p[y == 1] > thr).mean())


def prediction_metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return dict(brier=float(((p - y) ** 2).mean()),
                nll=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                ece=float(ece(p, y)), auroc=float(auroc(p, y)), recall_fpr5=recall_at_fpr(p, y))


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """IC 95 % de mean(a - b) con remuestreo pareado por episodio."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if len(d) == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(d)))


def summarize(rows, extras):
    """rows: lista de dicts por episodio de un mismo método; extras: arrays de tiempos y predicción."""
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    out = {}
    out["V_any"] = rate(col("V_any"))
    for k in ("dist_link", "dist_palm_body", "force", "unauth_contact"):
        out[f"V_{k}"] = rate(col(f"V_{k}"))
    out["S_force_mean"] = float(col("S_force").mean())
    out["S_link_mean"] = float(col("S_dist_link").mean())
    out["worst_link_deficit_p95"] = float(np.percentile(col("worst_link_deficit"), 95))
    out["peak_F_mean"] = float(col("peak_F").mean())
    out["peak_F_p95"] = float(np.percentile(col("peak_F"), 95))
    out["impulse_over_mean"] = float(col("impulse_over").mean())
    rel = [r for r in rows if r["release_applicable"]]
    rdt = np.array([r["release_dt"] for r in rel], float) * 1e3
    done = rdt[~np.isnan(rdt)]
    out["release_n"] = len(rel)
    out["release_ms_median"] = float(np.median(done)) if len(done) else np.nan
    out["release_ms_p95"] = float(np.percentile(done, 95)) if len(done) else np.nan
    out["release_censored"] = float(np.isnan(rdt).mean()) if len(rel) else np.nan
    out["release_fail"] = rate([r["release_fail"] for r in rel])
    out["false_cancel"] = rate(col("false_cancel"))
    out["success_auth"] = rate([r["success"] for r in rows if r["authorized"] and not r["withdraw"]])
    out["nocontact_unauth"] = rate([r["success"] for r in rows if not r["authorized"]])
    nqp, ninf = col("n_qp").sum(), col("n_infeas").sum()
    out["infeas_cycles"] = float(ninf / nqp) if nqp else np.nan
    out["infeas_episodes"] = rate(col("n_infeas") > 0)
    out["J_int_mean"] = float(col("J_int").mean())
    out["jerk_p95_mean"] = float(col("jerk_p95").mean())
    out["Fdot_p95_mean"] = float(col("Fdot_p95").mean())
    ts = np.concatenate([e["t_safety"] for e in extras])
    tq = np.concatenate([e["t_qp"] for e in extras]) if any(len(e["t_qp"]) for e in extras) else np.zeros(1)
    th = np.concatenate([e["t_hmm"] for e in extras])
    out["lat_p50"], out["lat_p95"], out["lat_p99"] = [float(np.percentile(ts, q)) for q in (50, 95, 99)]
    out["lat_max"] = float(ts.max())
    out["lat_over_budget"] = float((ts > 150.0).mean())
    out["qp_p50"], out["qp_p99"] = float(np.percentile(tq, 50)), float(np.percentile(tq, 99))
    out["hmm_p50"] = float(np.percentile(th, 50)) if len(th) else np.nan
    out["pred"] = prediction_metrics(np.concatenate([e["pred_p"] for e in extras]),
                                     np.concatenate([e["pred_y"] for e in extras]))
    return out


def cluster_paired_bootstrap(a, b, clusters, n_boot=4000, seed=0):
    """Diferencia media de a - b (pareada por episodio) con remuestreo por clúster (participante).
    Devuelve IC 95 % percentil y p-valor bilateral bootstrap."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    m = ~(np.isnan(a) | np.isnan(b))
    d, c = a[m] - b[m], c[m]
    if len(d) == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=0, n_clusters=0)
    uc = np.unique(c)
    sums = np.array([d[c == k].sum() for k in uc])
    cnts = np.array([(c == k).sum() for k in uc])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uc), (n_boot, len(uc)))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    p = max(p, 1.0 / n_boot) if d.mean() != 0 or (bs != 0).any() else 1.0
    return dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p=float(min(p, 1.0)), n=int(len(d)), n_clusters=int(len(uc)))


def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj
