"""Núcleo de decisión compartido por el simulador planar y el G1 en MuJoCo.

Todos los métodos (B0-B9, A1-A11) pasan por las MISMAS funciones: `method_signals` (regla de
riesgo/márgenes) y `safety_qp` (filtro). Un método se distingue sólo por los campos de `Method`,
lo que permite atribuir diferencias a mecanismos y no a implementaciones distintas.
"""
from dataclasses import dataclass, replace

import numpy as np
import quadprog

from . import params as P
from .belief import group_of

E, H, W = 0, 1, 2
MAP_TABLE = np.array([[0.0, 0.0], [0.05, 3.0], [0.10, 6.0]])


@dataclass(frozen=True)
class Method:
    name: str
    filter: bool = True
    risk: str = "none"           # none | map | entropy | bcbf | crc | rcip | pabe
    theta_update: str = "fixed"  # fixed | instant | exo | governor | sequential
    unc: str = "exc"             # (pabe) exc | global | none
    p_source: str = "upper"      # (pabe) upper | upper_bins | raw | none
    per_link: bool = True
    tactile: bool = True
    stochastic: bool = False     # SHIELD
    prudence: bool = False
    hsmm: bool = False           # A6: duración explícita del estado H
    tau: float = None            # umbral de liberación semántica
    alpha: float = None          # nivel de RCIP (B8)
    kappa: float = None          # ganancia del filtro exógeno de theta (B6) / referencia del gobernador
    rho_mult: float = 1.0        # A9: rho -> inf (el gobernador sólo se aparta de la referencia si es inviable)
    mix_h: bool = False          # A11: regla de liberación r = p + 0,4 H (como B3) con el resto de B9
    label: str = ""


_B9 = Method("B9", risk="pabe", theta_update="governor", prudence=True, label="PABe-CBF (completo)")
METHODS = {
    "B0": Method("B0", filter=False, label="Nominal sin filtro (solo sim.)"),
    "B1": Method("B1", label="CBF híbrido fijo"),
    "B2": Method("B2", risk="map", theta_update="instant", label="HMM-MAP + CBF"),
    "B3": Method("B3", risk="entropy", theta_update="instant", label="HMM-H(b) + CBF"),
    "B4": Method("B4", risk="bcbf", theta_update="instant", label="BCBF (Han et al.) adapt."),
    "B5": Method("B5", risk="crc", theta_update="instant", label="CRC-CBF adapt."),
    "B6": Method("B6", risk="pabe", theta_update="exo", prudence=True, label="TV-CBF robusto adapt."),
    "B7": Method("B7", stochastic=True, label="SHIELD adapt."),
    "B8": Method("B8", risk="rcip", label="RCIP + CBF fijo"),
    "B9": _B9,
    "A1": replace(_B9, name="A1", theta_update="instant", label="B9 sin gobernador (theta instantáneo)"),
    "A2": replace(_B9, name="A2", unc="global", label="B9 con H(b) global en vez de E_exc"),
    "A3": replace(_B9, name="A3", p_source="none", label="B9 sin p_W^+"),
    "A4": replace(_B9, name="A4", p_source="raw", label="B9 con b_W crudo (sin calibración)"),
    "A5": replace(_B9, name="A5", per_link=False, label="B9 con barrera geométrica única"),
    "A6": replace(_B9, name="A6", hsmm=True, label="B9 con HSMM (duración explícita)"),
    "A7": replace(_B9, name="A7", tactile=False, label="B9 sin modalidad táctil"),
    "A8": replace(_B9, name="A8", p_source="upper_bins", label="B9 con p_W^+ por bins (v1)"),
    "A9": replace(_B9, name="A9", rho_mult=1e6, label="B9 con rho->inf (sólo corrige por viabilidad)"),
    "A10": replace(_B9, name="A10", theta_update="sequential", label="B9 secuencial (theta y u por separado)"),
    "A11": replace(_B9, name="A11", mix_h=True, label="B9 con regla de liberación p+0,4H (como B3)"),
}


def method_from_spec(spec):
    """'B9', 'B9@0.9' (tau), 'B6@0.9@k8' (tau, kappa), 'B8@a0.1' (alpha)."""
    parts = spec.split("@")
    m = METHODS[parts[0]]
    kw = {}
    for p in parts[1:]:
        if p.startswith("k"):
            kw["kappa"] = float(p[1:]) if p[1:] != "inf" else np.inf
        elif p.startswith("a"):
            kw["alpha"] = float(p[1:])
        else:
            kw["tau"] = float(p)
    return replace(m, name=spec, **kw) if kw else m


@dataclass
class Models:
    A: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    prior: np.ndarray
    href: object
    calib: object
    crc_q: np.ndarray
    rcip_q: float
    calib_bins: object = None
    rcip_scores: np.ndarray = None
    hsmm: object = None          # parámetros del HSMM (A6)
    crc_err: dict = None         # errores de predicción por contexto (B5, G1)
    w_design: float = None       # cota de perturbación identificada (G1)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def rcip_q(md, alpha):
    if alpha is None or md.rcip_scores is None:
        return md.rcip_q
    s = np.sort(md.rcip_scores)
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


_CRC_CACHE = {}


def _crc_quantile(errs, alpha):
    k = (id(errs), alpha)
    if k not in _CRC_CACHE:
        s = np.sort(errs)
        i = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
        _CRC_CACHE[k] = float(s[min(max(i, 0), len(s) - 1)])
    return _CRC_CACHE[k]


def method_signals(m, md, rng, b, Hn, Eexc, mode, ctx):
    """theta*, liberación semántica, puerta A->C, pausa (RCIP), probabilidad reportada, prudencia."""
    bW = float(b[W])
    out = dict(theta_star=P.THETA0.copy(), sem=False, gate=True, pause=False, p=bW, prud=1.0)
    if m.risk == "map":
        z = int(np.argmax(b))
        out.update(theta_star=MAP_TABLE[z].copy(), sem=z == W, gate=z == E)
    elif m.risk == "entropy":
        r = bW + 0.4 * Hn
        out.update(theta_star=np.clip(P.THETA0 + P.K_R * bW + P.K_E * Hn, 0, P.THETA_MAX),
                   sem=r > (m.tau or 0.6), gate=r < min(0.5, m.tau or 0.6))
    elif m.risk == "pabe":
        p = {"upper": md.calib.upper_of(bW),
             "upper_bins": md.calib_bins.upper_of(bW) if md.calib_bins else 1.0,
             "raw": bW, "none": 0.0}[m.p_source]
        U = {"exc": Eexc, "global": Hn, "none": 0.0}[m.unc]
        out.update(theta_star=np.clip(P.THETA0 + P.K_R * p + P.K_E * U, 0, P.THETA_MAX),
                   sem=(m.p_source != "none") and (p + (0.4 * Hn if m.mix_h else 0.0)) > (m.tau or P.TAU_REL_PABE),
                   gate=(p + (0.4 * Hn if m.mix_h else 0.0)) < min(P.TAU_GATE_PABE, m.tau or P.TAU_REL_PABE),
                   p=p if m.p_source != "none" else bW,
                   prud=max(0.3, 1.0 - P.C_PRUDENCE * U) if m.prudence else 1.0)
    elif m.risk == "crc":
        pm = md.calib.mean_of(bW)
        qe = md.crc_q[group_of(mode)]
        if np.ndim(qe) == 1:                 # G1: contexto adicional por velocidad de la mano humana
            vv = int(np.linalg.norm(ctx["vh"]) > 0.15)
            qe = qe[vv]
            errs = getattr(md, "crc_err", None)
            if m.alpha is not None and errs is not None:      # nivel alpha elegido en validación
                qe = _crc_quantile(errs[(group_of(mode), vv)], m.alpha)
        out.update(theta_star=np.clip(np.array([qe, P.K_C * qe * 0.3]), 0, P.THETA_MAX),
                   sem=pm > (m.tau or 0.5), gate=pm < min(0.3, m.tau or 0.5), p=pm)
    elif m.risk == "bcbf":
        # creencia muestreada: posición futura de la mano bajo la mezcla de modelos por estado latente
        N, Tp, beta = 128, 0.25, 0.2
        dim = len(ctx["h"])
        zs = rng.choice(3, N, p=b / b.sum())
        v = np.empty((N, dim))
        nE, nH, nW = (zs == E).sum(), (zs == H).sum(), (zs == W).sum()
        v[zs == E] = ctx["vh"] + rng.normal(0, 0.15, (nE, dim))
        v[zs == H] = rng.normal(0, 0.15, (nH, dim))
        v[zs == W] = unit(ctx["rest"] - ctx["h"]) * rng.uniform(0.3, 1.2, (nW, 1))
        hp = ctx["h"] + v * Tp
        ds = np.linalg.norm(ctx["palm"] - hp, axis=1)
        eps_n = np.sqrt(np.log(2 / 0.05) / (2 * N))           # corrección de concentración (DKW)
        qlo = np.quantile(ds, max(beta - eps_n, 0.0))
        qhi = np.quantile(P.K_C * (ds - P.R_C), min(1 - beta + eps_n, 1.0))
        th_d = max(0.0, np.linalg.norm(ctx["palm"] - ctx["h"]) - qlo)
        if ctx.get("contact_mode_margins") and mode == "C":
            th_d = 0.0          # en C la separación palma-mano no está activa: rige la cola de fuerza (theta_F)
        th = np.array([th_d, max(0.0, qhi - ctx["F_hat"]) if ctx["engaged"] else 0.0])
        out.update(theta_star=np.clip(th, 0, P.THETA_MAX), sem=bW > (m.tau or 0.5),
                   gate=bW < min(0.3, m.tau or 0.5))
    elif m.risk == "rcip":
        q = rcip_q(md, m.alpha)
        S = {i for i in range(3) if 1.0 - b[i] <= q}
        out.update(sem=(W in S and E not in S), gate=(S == {E}),
                   pause=(S != {E}) and not (W in S and E not in S))
    return out


def _solve(G, a, C, b):
    try:
        return quadprog.solve_qp(G, a, C, b, 0)[0]
    except ValueError:
        return None


def safety_qp(rows, u_nom, theta, theta_star, mode_theta, theta_dot_exo, u_max, dt,
              delta_override=None, rho_mult=1.0, kappa=None):
    """QP (Ec. 9). rows: (h, a_u, a_th, exo, delta, gamma) con a_u u + a_th v + exo >= -gamma h + delta.
    mode_theta: 'governor' (u y v_theta conjuntos), 'sequential' (primero v_theta con u=u_nom, luego u),
    otro (theta exógeno con derivada theta_dot_exo). Devuelve (u, v_theta, info)."""
    nu = len(u_nom)
    if isinstance(u_max, tuple):            # límites por componente (p. ej. caja de aceleración)
        u_lo, u_hi = u_max
    else:
        u_lo, u_hi = np.full(nu, -u_max), np.full(nu, u_max)
    kap = P.KAPPA if kappa is None else kappa
    lo_v = np.maximum(-P.V_THETA_MAX, -theta / dt)
    hi_v = np.minimum(P.V_THETA_MAX, (P.THETA_MAX - theta) / dt)
    v_ref = np.clip(kap * (theta_star - theta), lo_v, hi_v) if np.isfinite(kap) else \
        np.clip((theta_star - theta) / dt, lo_v, hi_v)
    deltas = [r[4] if delta_override is None else delta_override[i] for i, r in enumerate(rows)]
    info = dict(v_ref=v_ref, gov_active=False)

    def solve(var_u, var_v, v_fixed=None, u_fixed=None):
        n = (nu if var_u else 0) + (2 if var_v else 0)
        Cs, bs = [], []
        for (h, a_u, a_th, exo, _, gam), dl in zip(rows, deltas):
            rhs = -gam * h + dl - exo
            row = []
            if var_u:
                row.append(a_u)
            else:
                rhs -= a_u @ u_fixed
            if var_v:
                row.append(a_th)
            else:
                rhs -= a_th @ v_fixed
            Cs.append(np.concatenate(row)); bs.append(rhs)
        lo = np.concatenate(([u_lo] if var_u else []) + ([lo_v] if var_v else []))
        hi = np.concatenate(([u_hi] if var_u else []) + ([hi_v] if var_v else []))
        I = np.eye(n)
        for i in range(n):
            Cs.append(I[i]); bs.append(lo[i]); Cs.append(-I[i]); bs.append(-hi[i])
        wts = np.concatenate(([P.R_U[0] * np.ones(nu)] if var_u else []) + ([P.RHO * rho_mult] if var_v else []))
        ref = np.concatenate(([u_nom] if var_u else []) + ([v_ref] if var_v else []))
        x = _solve(np.diag(2 * wts), 2 * wts * ref, np.array(Cs).T, np.array(bs))
        return x

    if mode_theta == "governor":
        x = solve(True, True)
        if x is None:
            return None, None, info
        u, v = x[:nu], x[nu:]
        info["gov_active"] = bool(np.abs(v - v_ref).max() > 1e-3 * (1 + np.abs(v_ref).max()))
        return u, v, info
    if mode_theta == "sequential":
        # 1) el margen avanza lo que sea viable SIN modificar la acción nominal; 2) se filtra u con ese v fijo
        v = solve(False, True, u_fixed=u_nom)
        if v is None:
            v = np.clip(np.zeros(2), lo_v, hi_v)
        x = solve(True, False, v_fixed=v)
        if x is None:
            return None, None, info
        info["gov_active"] = bool(np.abs(v - v_ref).max() > 1e-3 * (1 + np.abs(v_ref).max()))
        return x, v, info
    x = solve(True, False, v_fixed=theta_dot_exo)
    return (x, None, info) if x is not None else (None, None, info)
