"""Bucle de episodio: percepción -> HMM -> señales de riesgo -> autómata A/C/R -> QP CBF -> robot.

Implementa B0..B9 (Tabla 2) y las ablaciones A1..A5, A7 (Tabla 4) sobre una abstracción planar
del apretón de manos. Todos los métodos comparten controlador nominal, sensores, geometría,
restricciones físicas, permisos de contacto y política de respaldo; sólo cambia la regla de
riesgo/márgenes y el modo de actualizarlos (Sec. 7.1).
"""
import time
from dataclasses import dataclass, replace

import numpy as np

from . import params as P
from .belief import ModeHMM, entropy, group_of
from .kinematics import fk, dls, Q_HOME, human_points, ROBOT_POINTS, HUMAN_POINTS
from .scenarios import E, H, W

W_DESIGN = 0.05                       # cota de perturbación asumida por el diseño robusto (A2)
DIR_APPROACH = np.array([-1.0, -0.4]) / np.linalg.norm([-1.0, -0.4])
MAP_TABLE = np.array([[0.0, 0.0], [0.05, 3.0], [0.10, 6.0]])
LINK_PTS = ("elbow", "fmid", "wrist")


from .core import (Method, METHODS, Models, method_from_spec, method_signals, safety_qp,  # noqa: F401
                   unit as _unit, MAP_TABLE)


# ------------------------------------------------------------------------------------
# Barreras: cada fila (h, a_u(3), a_th(2), exo, delta, gamma) representa
#   a_u u + a_th v_theta + exo >= -gamma h + delta          (Ec. 9)
# ------------------------------------------------------------------------------------
def _dist_row(p, J, hk, vk, d_phys, c_theta, eps_d, eps_v, eps_growth, w_bound):
    d = np.linalg.norm(p - hk)
    n = (p - hk) / max(d, 1e-9)
    a_u = n @ J
    h = d - d_phys - eps_d - c_theta[0] - P.SAMPLE_MARGIN_D
    exo = -n @ vk - eps_v - eps_growth
    return [h, a_u, np.array([-c_theta[1], 0.0]), exo, w_bound * np.abs(a_u).sum(), P.GAMMA_D]


def build_barriers(mode, engaged, pts, hum, hvel, est, theta, m, rhoR, rhoR_dot, w_bound):
    """est: dict con eps_d, eps_v, growth (crecimiento de la cota durante oclusión) y F_hat."""
    rows = []
    eps_d, eps_v, gr = est["eps_d"], est["eps_v"], est["growth"]
    th_d = theta[0]
    cl = P.LINK_THETA_GAIN
    palm, Jp = pts["palm"]
    if m.per_link:
        for r in LINK_PTS:
            p, J = pts[r]
            for k in HUMAN_POINTS:
                st = k == "torso"
                rows.append(_dist_row(p, J, hum[k], hvel[k], P.D_PHYS_LINK, (cl * th_d, cl),
                                      0.0 if st else eps_d, 0.0 if st else eps_v, 0.0 if st else gr, w_bound))
        for k in ("hwrist", "helbow", "torso"):
            st = k == "torso"
            rows.append(_dist_row(palm, Jp, hum[k], hvel[k], P.D_PHYS_PALM_BODY, (cl * th_d, cl),
                                  0.0 if st else eps_d, 0.0 if st else eps_v, 0.0 if st else gr, w_bound))
    hk, vk = hum["hand"], hvel["hand"]
    if mode == "A":
        rows.append(_dist_row(palm, Jp, hk, vk, P.D_SEP_PALM, (th_d, 1.0), eps_d, eps_v, gr, w_bound))
    elif mode == "R" and not engaged:
        r = _dist_row(palm, Jp, hk, vk, 0.0, (th_d, 1.0), eps_d, eps_v, gr, w_bound)
        r[0] -= rhoR
        r[3] -= rhoR_dot
        rows.append(r)
    if mode == "C" or (mode == "R" and engaged):
        d = np.linalg.norm(palm - hk)
        n = (palm - hk) / max(d, 1e-9)
        nJ = n @ Jp
        Fs = est["F_hat"]                      # positivo = tirón, negativo = compresión
        for F_lim, a_thF in ((P.F_PHYS, 0.0), (P.F_SOC0, -1.0)):
            F_lim_eff = F_lim - (theta[1] if a_thF else 0.0)
            if engaged:
                # tirón: h = F_lim - eps_F - F_hat ; dF/dt = K_C (nJ u - n v_h)
                rows.append([F_lim_eff - P.EPS_F - Fs - P.SAMPLE_MARGIN_F, -P.K_C * nJ,
                             np.array([0.0, a_thF]), P.K_C * (n @ vk) - P.K_C * eps_v,
                             P.K_C * w_bound * np.abs(nJ).sum(), P.GAMMA_F])
                push = -Fs
            else:
                push = P.K_C * (P.R_C - d + eps_d)  # modelo geométrico conservador antes del agarre
            rows.append([F_lim_eff - P.EPS_F - push - P.SAMPLE_MARGIN_F, P.K_C * nJ,
                         np.array([0.0, a_thF]), -P.K_C * (n @ vk) - P.K_C * (eps_v + gr),
                         P.K_C * w_bound * np.abs(nJ).sum(), P.GAMMA_F])
    return rows


# ------------------------------------------------------------------------------------
class Episode:
    def __init__(self, method, scen, models, collect=False):
        self.m, self.scen, self.models, self.collect = method, scen, models, collect
        self.hmm = ModeHMM(models.A, models.mu, models.sd, models.prior) if models is not None else None
        self.rng = np.random.default_rng(list(scen.rng_seed) + [99])
        self.trace = None                      # lista opcional para depuración
        self.record = None                     # lista opcional de estados por paso (figuras)

    # señales de riesgo por método -> theta*, liberación semántica, puerta A->C, pausa, p reportada
    def signals(self, b, Hn, Eexc, mode, ctx):
        ctx = dict(ctx, rest=self.scen.part.rest)
        return method_signals(self.m, self.models, self.rng, b, Hn, Eexc, mode, ctx)

    def nominal(self, mode, q, pts, h_est, vh_est, theta, engaged, anchor, prud, pause):
        palm, J = pts["palm"]
        if mode == "A":
            target = h_est + (P.D_SEP_PALM + P.EPS_D + theta[0] + 0.02) * DIR_APPROACH
            v = P.KP_NOM * (target - palm)
            vmax = P.V_NOM_MAX * prud
            if pause:
                vmax = 0.0
        elif mode == "C":
            if engaged:
                target, ff = anchor, np.zeros(2)          # guion: sostener el punto de agarre
            else:
                target, ff = h_est + P.R_C * _unit(palm - h_est), vh_est
            v = P.KP_NOM * (target - palm) + ff
            vmax = 1.0
        else:
            v = P.KP_NOM * (P.HOME_PALM - palm)
            vmax = P.V_NOM_MAX
        nv = np.linalg.norm(v)
        if nv > vmax:
            v = v * vmax / max(nv, 1e-9)
        u = dls(J, v)
        Jp = np.linalg.pinv(J)
        u = u + P.K_NULL * (np.eye(3) - Jp @ J) @ (Q_HOME - q)
        return np.clip(u, -P.U_MAX, P.U_MAX)

    def run(self):
        m, sc = self.m, self.scen
        dt = P.DT
        n_steps = int(P.T_MAX / dt)
        q = Q_HOME.copy()
        mode, t_mode = "A", 0.0
        theta = P.THETA0.copy()
        sig = dict(theta_star=P.THETA0.copy(), sem=False, gate=True, pause=False, p=0.02, prud=1.0)
        engaged, t_eng, anchor = False, None, None
        t_R, R_reason, rho0, grip_open_at, t_open = None, None, 0.0, None, None
        t_closure = None
        sem_count, pause_time = 0, 0.0
        b = self.hmm.prior.copy() if self.hmm else np.array([0.9, 0.08, 0.02])
        Hn, Eexc, z_prev, g_prev = entropy(b), 0.0, -1, -1
        u_buf = [np.zeros(3)] * (sc.act_delay + 1)
        res_mean, res_var = np.zeros(3), np.full(3, 1e-4)
        q_prev, u_applied_prev = None, None
        F_hist = []
        h_est_hold, vh_hold, last_seen = None, None, 0.0
        delay_steps = int(round(sc.tactile_delay / dt))

        # registros
        palm_log, F_log = [], []
        viol = dict(dist_link=False, dist_palm_body=False, force=False, unauth_contact=False)
        sev = dict(dist_link=0.0, dist_palm_body=0.0, force=0.0)
        worst_link = 0.0
        peak_F, impulse, impulse_over = 0.0, 0.0, 0.0
        J_int = 0.0
        n_qp, n_infeas = 0, 0
        t_safety, t_hmm, t_qp = [], [], []
        pred_p, pred_y, pred_fam = [], [], []
        collect = dict(obs=[], H=[], cal=[], crc=[], rcip=[]) if self.collect else None
        touched_hand = False
        unauth_info = None
        rel_done_t, rel_fail_force, contact_after_onset = None, False, False
        F = 0.0

        for k in range(n_steps):
            t = k * dt
            # ---------------- verdad del terreno ----------------
            h = sc.hand(t)
            hp = human_points(h)
            hp2 = human_points(sc.hand(t + dt))
            hv = {key: (hp2[key] - hp[key]) / dt for key in HUMAN_POINTS}
            pts = fk(q)
            palm = pts["palm"][0]
            d_ph = np.linalg.norm(palm - h)
            F = P.K_C * (d_ph - P.R_C) if engaged else -P.K_C * max(0.0, P.R_C - d_ph)
            z_true = sc.z(t)

            # violaciones (Ec. 13) con magnitudes verdaderas
            for r in LINK_PTS:
                for key in HUMAN_POINTS:
                    zz = (P.D_PHYS_LINK - np.linalg.norm(pts[r][0] - hp[key])) / P.D_PHYS_LINK
                    worst_link = max(worst_link, zz)
                    if zz > 0:
                        viol["dist_link"] = True; sev["dist_link"] += zz * dt
                    if zz > 0.5:
                        viol["unauth_contact"] = True   # eslabón no autorizado a <3 cm: contacto
                        unauth_info = unauth_info or (round(t, 2), mode, r, key)
            for key in ("hwrist", "helbow", "torso"):
                zz = (P.D_PHYS_PALM_BODY - np.linalg.norm(palm - hp[key])) / P.D_PHYS_PALM_BODY
                if zz > 0:
                    viol["dist_palm_body"] = True; sev["dist_palm_body"] += zz * dt
            zf = (abs(F) - P.F_PHYS) / P.F_PHYS
            if zf > 0:
                viol["force"] = True; sev["force"] += zf * dt
                impulse_over += (abs(F) - P.F_PHYS) * dt
            peak_F = max(peak_F, abs(F))
            impulse += abs(F) * dt
            touching = d_ph <= P.R_C + P.TOUCH
            if touching:
                touched_hand = True
                authorized_ctx = sc.authorized and (mode == "C" or (mode == "R" and t - t_R <= 0.5))
                if not authorized_ctx:
                    viol["unauth_contact"] = True
                    unauth_info = unauth_info or (round(t, 2), mode, "palm", "hand")
            if sc.t_W is not None and t >= sc.t_W:
                if engaged or touching:
                    contact_after_onset = True
                if zf > 0:
                    rel_fail_force = True
                if rel_done_t is None and contact_after_onset and not engaged and abs(F) < 1.0 \
                        and d_ph >= P.D_SEP_PALM:
                    rel_done_t = t
            palm_log.append(palm.copy()); F_log.append(F)

            # ---------------- percepción ----------------
            tic = time.perf_counter()
            visible = sc.visible(t)
            if visible:
                off = sc.noise_pos[k]
                h_est = {key: (hp[key] + off if key != "torso" else hp[key]) for key in HUMAN_POINTS}
                vh_est = {key: (hv[key] + sc.noise_vel[k] if key != "torso" else np.zeros(2)) for key in HUMAN_POINTS}
                h_est_hold, vh_hold, last_seen = h_est, vh_est, t
                est = dict(eps_d=P.EPS_D, eps_v=P.EPS_V, growth=0.0)
            else:
                # oclusión: última posición retenida y conjunto alcanzable con velocidad inicial
                # conocida y aceleración acotada: eps(s) = eps_d + (|v|+eps_v) s + a_max s^2 / 2
                s_ = t - last_seen
                v0 = np.linalg.norm(vh_hold["hand"]) + P.EPS_V
                h_est = h_est_hold
                vh_est = {key: np.zeros(2) for key in HUMAN_POINTS}
                est = dict(eps_d=P.EPS_D + v0 * s_ + 0.5 * P.A_MAX_H * s_**2, eps_v=0.0,
                           growth=v0 + P.A_MAX_H * s_)
            F_hat = F + sc.noise_F[k]
            F_hist.append(F_hat)
            F_tact = F_hist[k - delay_steps] if k >= delay_steps else 0.0
            est["F_hat"] = F_hat
            dhat_ph = np.linalg.norm(palm - h_est["hand"])

            # ---------------- HMM (lazo lento) ----------------
            if k % P.HMM_EVERY == 0:
                th0 = time.perf_counter()
                tick = k // P.HMM_EVERY
                g = group_of(mode)
                y = sc.observe(tick, t, g, t_eng if engaged else None, tactile=m.tactile)
                if collect is not None:
                    collect["obs"].append((g, z_true, y, z_prev if g == g_prev else -1))
                    z_prev, g_prev = z_true, g
            if self.hmm is not None and k % P.HMM_EVERY == 0:
                b, bbar, Hn, Sp = self.hmm.step(y, mode)
                Eexc = max(0.0, Hn - self.models.href(mode, t - t_mode)) if self.models.href else 0.0
                ctx = dict(h=h_est["hand"], vh=vh_est["hand"], palm=palm, F_hat=F_hat, engaged=engaged)
                sig = self.signals(b, Hn, Eexc, mode, ctx)
                sem_count = sem_count + 1 if sig["sem"] else 0
                t_hmm.append((time.perf_counter() - th0) * 1e3)
                if t_closure is None:
                    pred_p.append(sig["p"]); pred_y.append(int(z_true == W))
                if collect is not None and mode != "R" and t_closure is None:   # ticks relevantes para decidir
                    collect["H"].append((g, t - t_mode, Hn, sc.family, sc.t_W is None))
                    collect["cal"].append((float(b[W]), int(z_true == W)))
                    collect["rcip"].append(1.0 - float(b[z_true]))
                    Tp = 0.25
                    if visible and t + Tp < P.T_MAX:
                        pred = h_est["hand"] + vh_est["hand"] * Tp
                        collect["crc"].append((g, float(np.linalg.norm(sc.hand(t + Tp) - pred))))
            release_sem = sem_count >= P.PERSIST_TICKS

            # ---------------- autómata de fase A/C/R ----------------
            new_mode = None
            if mode == "A":
                pause_time = pause_time + dt if sig["pause"] else 0.0   # pausa consecutiva (petición de aclaración)
                if release_sem or pause_time > 1.5:
                    new_mode, R_reason = "R", "abort"
                else:
                    auth_obs = sc.authorized and t >= sc.t_auth + 0.1
                    r_ready = P.D_SEP_PALM + est["eps_d"] + theta[0] + 0.05
                    if auth_obs and sig["gate"] and dhat_ph <= r_ready:
                        rows_c = build_barriers("C", engaged, pts, h_est, vh_est, est, theta, m, 0, 0, W_DESIGN)
                        if all(r[0] >= 0 for r in rows_c):           # entrada segura en C_{m+}
                            new_mode = "C"
            elif mode == "C":
                if abs(F_hat) > P.F_HW:
                    new_mode, R_reason = "R", "hw"
                elif m.tactile and F_tact > P.F_PULL_EXPLICIT:
                    new_mode, R_reason = "R", "pull"
                elif release_sem:
                    new_mode, R_reason = "R", "sem"
                elif engaged and t - t_eng >= P.T_SHAKE:
                    new_mode, R_reason = "R", "closure"
                    t_closure = t
                    sc.closure(t)
            if new_mode is not None:
                mode, t_mode = new_mode, t
                sem_count = 0
                if mode == "R":
                    t_R = t
                    grip_open_at = t + P.GRIP_DELAY
            rhoR, rhoR_dot = 0.0, 0.0
            if mode == "R" and not engaged:
                if t_open is None:      # la rampa de separación arranca al abrir la mano (entrada segura)
                    t_open = t
                    rho0 = max(0.0, dhat_ph - est["eps_d"] - theta[0] - P.SAMPLE_MARGIN_D - 0.005)
                rhoR = min(P.D_SEP_PALM, rho0 + 0.3 * (t - t_open))
                rhoR_dot = 0.3 if rhoR < P.D_SEP_PALM else 0.0

            # ---------------- márgenes theta ----------------
            theta_star = sig["theta_star"]
            theta_dot_exo = np.zeros(2)
            if m.theta_update == "instant":
                theta = theta_star.copy()
            elif m.theta_update == "exo":
                theta_dot_exo = np.clip((m.kappa or P.KAPPA) * (theta_star - theta), -P.V_THETA_MAX, P.V_THETA_MAX)
            elif m.theta_update == "fixed":
                theta = P.THETA0.copy()

            # ---------------- nominal + filtro ----------------
            u_nom = self.nominal(mode, q, pts, h_est["hand"], vh_est["hand"], theta, engaged, anchor,
                                 sig["prud"] if mode == "A" else 1.0, sig["pause"] and mode == "A")
            v_theta = None
            if m.filter:
                rows = build_barriers(mode, engaged, pts, h_est, vh_est, est, theta, m, rhoR, rhoR_dot, W_DESIGN)
                delta_o = None
                if m.stochastic:
                    # SHIELD: E[h_{k+1}] >= (1-gamma dt) h_k con residual de dinámica aprendido en línea
                    sd = np.sqrt(res_var)
                    delta_o = []
                    for r in rows:
                        grad = r[1]                                    # dh/dq
                        delta_o.append(-grad @ res_mean + 1.0 * np.linalg.norm(grad * sd))
                tq = time.perf_counter()
                u, v_theta, _ = safety_qp(rows, u_nom, theta, theta_star, m.theta_update, theta_dot_exo,
                                          P.U_MAX, P.DT, delta_o, m.rho_mult, m.kappa)
                t_qp.append((time.perf_counter() - tq) * 1e3)
                n_qp += 1
                if self.trace is not None:
                    hs = [r[0] for r in rows]
                    self.trace.append((round(t, 2), mode, engaged, round(d_ph, 3), round(F, 1), u is None,
                                       int(np.argmin(hs)), round(min(hs), 3), theta.round(3).tolist()))
                if u is None:
                    n_infeas += 1
                    # respaldo: con agarre cerrado acompaña la mano (liberación controlada); si no, se aleja
                    v_b = vh_est["hand"] if engaged else 0.5 * _unit(palm - h_est["hand"])
                    u = np.clip(dls(pts["palm"][1], v_b), -P.U_MAX, P.U_MAX)
                    v_theta = np.zeros(2) if m.theta_update in ("governor", "sequential") else None
                    if mode == "C":
                        mode, t_mode, R_reason, t_R = "R", t, "backup", t
                        grip_open_at = t + P.GRIP_DELAY
                J_int += float((((u - u_nom) / P.U_MAX) ** 2).sum() * dt)
            else:
                u = u_nom
            t_safety.append((time.perf_counter() - tic) * 1e3)

            if self.record is not None:
                self.record.append(dict(
                    t=t, q=q.copy(), pts={k_: pts[k_][0].copy() for k_ in ROBOT_POINTS},
                    hum={k_: hp[k_].copy() for k_ in HUMAN_POINTS}, mode=mode, engaged=engaged,
                    theta=theta.copy(), theta_star=np.array(theta_star, float).copy(), F=F, d=d_ph,
                    b=np.array(b, float).copy(), p=sig["p"], H=Hn, Eexc=Eexc, z=z_true,
                    u=np.array(u, float).copy(), u_nom=np.array(u_nom, float).copy()))

            # ---------------- integración ----------------
            u_buf.append(u)
            u_app = u_buf[-1 - sc.act_delay]
            u_buf = u_buf[-(sc.act_delay + 2):]
            q_new = q + (u_app + sc.w[k]) * dt
            if m.stochastic:
                r_obs = (q_new - q) / dt - u_app
                res_mean = 0.98 * res_mean + 0.02 * r_obs
                res_var = 0.98 * res_var + 0.02 * (r_obs - res_mean) ** 2
            q = q_new
            if m.theta_update in ("governor", "sequential") and v_theta is not None:
                theta = np.clip(theta + v_theta * dt, 0, P.THETA_MAX)
            elif m.theta_update == "exo":
                theta = np.clip(theta + theta_dot_exo * dt, 0, P.THETA_MAX)

            # ---------------- agarre ----------------
            palm_n = fk(q)["palm"][0]
            d_n = np.linalg.norm(palm_n - sc.hand(t + dt))
            if mode == "C" and not engaged and d_n <= P.R_C + P.TOUCH:
                engaged, t_eng, anchor = True, t, palm_n.copy()
            if mode == "R" and engaged and t >= grip_open_at:
                engaged = False
            if engaged and P.K_C * abs(d_n - P.R_C) > P.F_BREAK:
                engaged = False                                     # deslizamiento

            # ---------------- fin de episodio ----------------
            if t_closure is not None and t >= t_closure + 1.2:
                break
            if mode == "R" and t_closure is None and t >= t_R + 1.5 and not engaged and d_ph >= P.D_SEP_PALM:
                break

        T_e = (k + 1) * dt
        palm_arr, F_arr = np.array(palm_log), np.array(F_log)
        ker = np.ones(5) / 5
        ps = np.stack([np.convolve(palm_arr[:, i], ker, mode="valid") for i in range(2)], 1)
        jerk = np.linalg.norm(np.diff(ps, 3, axis=0), axis=1) / dt**3 if len(ps) > 4 else np.zeros(1)
        Fdot = np.abs(np.diff(np.convolve(F_arr, ker, mode="valid"))) / dt
        authorized_nominal = sc.authorized and sc.t_W is None
        res = dict(
            method=m.name, family=sc.family, pid=sc.part.pid, seed=sc.rng_seed[2], shift=sc.part.shift,
            authorized=sc.authorized, withdraw=sc.t_W is not None, T=T_e,
            V_any=any(viol.values()), **{f"V_{k_}": v for k_, v in viol.items()},
            **{f"S_{k_}": v for k_, v in sev.items()}, worst_link_deficit=worst_link,
            peak_F=peak_F, impulse=impulse, impulse_over=impulse_over,
            success=(t_closure is not None) if authorized_nominal else (not touched_hand if not sc.authorized else np.nan),
            false_cancel=(R_reason not in (None, "closure")) if authorized_nominal else np.nan,
            R_reason=R_reason, t_closure=t_closure, t_R=t_R, t_W=sc.t_W,
            release_applicable=(sc.t_W is not None and contact_after_onset),
            release_dt=(rel_done_t - sc.t_W) if (rel_done_t is not None) else np.nan,
            release_fail=(sc.t_W is not None and contact_after_onset and (rel_done_t is None or rel_fail_force)),
            J_int=J_int, jerk_p95=float(np.percentile(jerk, 95)), Fdot_p95=float(np.percentile(Fdot, 95)),
            n_qp=n_qp, n_infeas=n_infeas, unauth_info=str(unauth_info),
        )
        extra = dict(pred_p=np.array(pred_p), pred_y=np.array(pred_y, dtype=np.int8),
                     t_safety=np.array(t_safety, dtype=np.float32), t_hmm=np.array(t_hmm, dtype=np.float32),
                     t_qp=np.array(t_qp, dtype=np.float32))
        return res, extra, collect
