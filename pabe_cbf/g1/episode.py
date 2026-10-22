"""Episodio de saludo con el Unitree G1 en MuJoCo bajo PABe-CBF y comparadores.

Lazo: física 500 Hz (servos de posición del G1) | filtro CBF 100 Hz | HMM 33 Hz.
El filtro produce velocidades articulares u (7 GDL del brazo derecho) que se integran como
referencia de los servos; el error de seguimiento real entra como perturbación acotada (W_DESIGN
identificada en entrenamiento). Todas las decisiones de riesgo y el QP se delegan en pabe_cbf.core.
"""
import os
import time
from dataclasses import replace

import mujoco
import numpy as np

from .. import params as P
from ..belief import ModeHMM, ModeHSMM, entropy, group_of
from ..core import method_signals, safety_qp, unit
from ..scenarios import E, H, W
from . import geometry as G
from .model import G1Index, FINGER_CLOSED, build, stand_qpos

DT, SUB = 0.01, 5                    # control 100 Hz, física 2 ms
PH_OFF = G.R_HAND + G.PALM_R - 0.04  # d_ph := distancia centro palma-mano - PH_OFF (R_C = 4 cm <-> agarre)
VIS_DELAY = 0.033                    # retardo nominal de percepción visual (cámara + inferencia)
EXT_FORCE_VI = 15.0
LINK_ROBOT = ("upperarm", "forearm", "wrist")
LINK_HUMAN = ("hand", "farm", "uarm", "torso", "head")
APPROACH = unit(np.array([-1.0, -0.35, 0.0]))
D_PHYS_ADJ = 0.03                    # en C: muñeca <-> mano humana (adyacente a la superficie autorizada)
SHIELD_C = 2.0                       # SHIELD: delta = -grad·media + c·||grad ⊙ desv|| del residual aprendido
A_MAX = 50.0                         # límite de aceleración articular del comando (rad/s^2), común a todos
W_CART0 = 0.135                      # cota cartesiana por defecto (p99,9 medido); se reidentifica al entrenar
GAMMA_F = 40.0                       # clase K de las barreras de fuerza en el G1 (ver texto)
RAMP_R = 0.15                        # velocidad de la rampa de separación tras abrir la mano (m/s)
W_JOINT = 0.30                       # cota articular (rad/s) para las barreras de límites articulares
KP_CT, KD_CT = 400.0, 40.0           # ganancias del lazo de par calculado (por articulación, normalizadas)
CRC_V = 0.15                         # B5: contexto de velocidad de la mano humana (m/s) para el cuantil conforme
H_FILT = 0.25                        # filtro de la estimación usada SOLO por el controlador nominal
_MODEL = None


MJB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "g1_pabe_scene.mjb")


def get_model():
    """Modelo compilado una sola vez y cacheado en binario MJB (compilar las mallas del G1 cuesta ~560 MB
    residentes por proceso; cargar el MJB, ~120 MB)."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MJB_PATH):
            m = build()
            tmp = MJB_PATH + f".{os.getpid()}.tmp"
            mujoco.mj_saveModel(m, tmp, None)
            os.replace(tmp, MJB_PATH)
        _MODEL = mujoco.MjModel.from_binary_path(MJB_PATH)
    return _MODEL


class G1Episode:
    def __init__(self, method, scen, models, collect=False, clamp=None, record=False):
        self.m, self.sc, self.collect, self.clamp = method, scen, collect, clamp
        self.models = models
        self.record = [] if record else None
        self.rng = np.random.default_rng(list(scen.rng_seed) + [99])
        if models is not None:
            if method.hsmm and models.hsmm is not None:
                base = ModeHMM(models.A, models.mu, models.sd, models.prior)
                self.hmm = ModeHSMM(base, models.hsmm["k"], models.hsmm["A_exp"])
                self.md = replace(models, calib=models.hsmm["calib"], href=models.hsmm["href"])
            else:
                self.hmm = ModeHMM(models.A, models.mu, models.sd, models.prior)
                self.md = models
            self.w_cart = models.w_design if models.w_design is not None else W_CART0
        else:
            self.hmm, self.md, self.w_cart = None, None, W_CART0

    # ---------------------------------------------------------------------------------
    def _robot(self, mm, d, ix):
        """Cápsulas del robot y jacobianos (traslacionales, columnas del brazo)."""
        jac = {}
        for k, (ba, bb, r) in ix.cap_bodies.items():
            for b in (ba, bb):
                if b not in jac:
                    jp = np.zeros((3, mm.nv))
                    mujoco.mj_jac(mm, d, jp, None, d.xpos[b], b)
                    jac[b] = jp[:, ix.arm_v]
        caps = {k: (d.xpos[ba].copy(), d.xpos[bb].copy(), r, jac[ba], jac[bb]) for k, (ba, bb, r) in ix.cap_bodies.items()}
        jp, jr = np.zeros((3, mm.nv)), np.zeros((3, mm.nv))
        palm = d.site_xpos[ix.palm_site].copy()
        mujoco.mj_jac(mm, d, jp, jr, palm, ix.palm)
        self._hand_axis = unit(palm - d.xpos[ix.palm])
        self._Jr = jr[:, ix.arm_v]
        return caps, palm, jp[:, ix.arm_v]

    def _dist_row(self, cap_r, cap_h, vel_h, d_phys, c_th, est, w):
        a, b, rr, Ja, Jb = cap_r
        ha, hb, rh = cap_h
        dist, s, t, c1, c2 = G.seg_seg(a, b, ha, hb)
        n = (c1 - c2) / max(dist, 1e-9)
        J = (1 - s) * Ja + s * Jb
        a_u = n @ J
        vh = (1 - t) * vel_h[0] + t * vel_h[1]
        hval = dist - rr - rh - d_phys - est["eps_d"] - c_th[0] - P.SAMPLE_MARGIN_D
        exo = -n @ vh - est["eps_v"] - est["growth"]
        return [hval, a_u, np.array([-c_th[1], 0.0]), exo, w, P.GAMMA_D]

    def barriers(self, mode, engaged, caps, palm, Jp, hcap, hvel, est, theta, q, ix, rhoR=0.0, rhoR_dot=0.0,
                 relax=None):
        """relax: (offsets por fila, t desde la apertura) -> en R, las filas que nacen con poca holgura junto a la
        zona de agarre recuperan su separación nominal mediante una rampa (entrada segura en C_R)."""
        m, rows, w = self.m, [], self.w_cart   # cota cartesiana del error de seguimiento (Ec. 9, delta_i)
        cl, th_d = P.LINK_THETA_GAIN, theta[0]
        if m.per_link:
            for rk in LINK_ROBOT:
                for hk in LINK_HUMAN:
                    st = hk in ("torso", "head")
                    e = dict(est, eps_d=est["eps_d"] if not st else est["tube"], eps_v=0.0 if st else est["eps_v"],
                             growth=0.0 if st else est["growth"])
                    adj = mode != "A" and rk == "wrist" and hk == "hand"
                    dp = D_PHYS_ADJ if adj else P.D_PHYS_LINK
                    # par adyacente a la superficie autorizada: sin margen social (igual para todos los métodos)
                    cth = (0.0, 0.0) if adj else (cl * th_d, cl)
                    rows.append(self._dist_row(caps[rk], hcap[hk], hvel[hk], dp, cth, e, w))
            pcap = (palm, palm, G.PALM_R, Jp, Jp)
            for hk in (("farm", "uarm", "torso", "head") if mode == "A" or (mode == "R" and not engaged)
                       else ("uarm", "torso", "head")):   # en C la muñeca/antebrazo distal humano es zona de agarre
                st = hk in ("torso", "head")
                e = dict(est, eps_d=est["eps_d"] if not st else est["tube"], eps_v=0.0 if st else est["eps_v"],
                         growth=0.0 if st else est["growth"])
                rows.append(self._dist_row(pcap, hcap[hk], hvel[hk], P.D_PHYS_PALM_BODY, (cl * th_d, cl), e, w))
        h_c = hcap["hand"][0]
        vh = hvel["hand"][0]
        dc = np.linalg.norm(palm - h_c)
        n = (palm - h_c) / max(dc, 1e-9)
        nJ = n @ Jp
        d_ph = dc - PH_OFF
        if mode == "A" or (mode == "R" and not engaged):
            r = [d_ph - (P.D_SEP_PALM if mode == "A" else 0.0) - est["eps_d"] - th_d - P.SAMPLE_MARGIN_D, nJ,
                 np.array([-1.0, 0.0]), -n @ vh - est["eps_v"] - est["growth"], w, P.GAMMA_D]
            if mode == "R":
                r[0] -= rhoR
                r[3] -= rhoR_dot
            rows.append(r)
        if mode == "C" or (mode == "R" and engaged):
            Fs = est["F_hat"]
            lims = ((P.F_PHYS, 0.0), (P.F_SOC0, -1.0)) if mode == "C" else ((P.F_PHYS, 0.0),)   # en R: sólo el límite físico
            for F_lim, a_thF in lims:
                F_eff = F_lim - (theta[1] if a_thF else 0.0)
                dlt = P.K_C * w
                if engaged:
                    rows.append([F_eff - P.EPS_F - Fs - P.SAMPLE_MARGIN_F, -P.K_C * nJ, np.array([0.0, a_thF]),
                                 P.K_C * (n @ vh) - P.K_C * est["eps_v"], dlt, GAMMA_F])
                    push = -Fs
                else:
                    push = P.K_C * (P.R_C - d_ph + est["eps_d"])
                rows.append([F_eff - P.EPS_F - push - P.SAMPLE_MARGIN_F, P.K_C * nJ, np.array([0.0, a_thF]),
                             -P.K_C * (n @ vh) - P.K_C * (est["eps_v"] + est["growth"]), dlt, GAMMA_F])
        if relax is not None:
            offs, s_open = relax
            for i, o in offs.items():
                if i < len(rows):
                    rem = max(0.0, o - RAMP_R * s_open)
                    if rem > 0:
                        rows[i][0] += rem
                        rows[i][3] -= RAMP_R
        # límites articulares (comunes a todos los métodos)
        I = np.eye(7)
        for i in range(7):
            rows.append([q[i] - ix.arm_lo[i] - 0.05, I[i], np.zeros(2), 0.0, W_JOINT, 5.0])
            rows.append([ix.arm_hi[i] - 0.05 - q[i], -I[i], np.zeros(2), 0.0, W_JOINT, 5.0])
        return rows

    def nominal(self, mode, q, q_home, palm, Jp, h_est, vh_est, theta, engaged, anchor, prud, pause, home_palm):
        if mode == "A":
            # el robot espera a una distancia que crece con la velocidad de la mano humana (saludo guiado por la persona)
            standoff = P.D_SEP_PALM + PH_OFF + P.EPS_D + theta[0] + 0.02 + 0.30 * np.linalg.norm(vh_est)
            target = h_est + standoff * APPROACH
            v, vmax = P.KP_NOM * (target - palm), P.V_NOM_MAX * prud * (0.0 if pause else 1.0)
        elif mode == "C":
            if engaged:
                v, vmax = P.KP_NOM * (anchor - palm), 1.0
            else:
                target = h_est + (P.R_C + PH_OFF) * unit(palm - h_est)
                v, vmax = P.KP_NOM * (target - palm) + vh_est, 1.0
        else:
            v, vmax = P.KP_NOM * (home_palm - palm), P.V_NOM_MAX
        nv = np.linalg.norm(v)
        if nv > vmax:
            v = v * vmax / max(nv, 1e-9)
        # orientación: el eje muñeca->palma apunta hacia la mano humana (A y C); peso menor que la posición
        if mode in ("A", "C"):
            w_rot = 0.4
            om = 3.0 * G.cross3(self._hand_axis, unit(h_est - palm))
            J6 = np.vstack([Jp, w_rot * self._Jr]); v6 = np.concatenate([v, w_rot * om])
        else:
            J6, v6 = Jp, v
        Jinv = J6.T @ np.linalg.inv(J6 @ J6.T + 0.05**2 * np.eye(len(v6)))
        u = Jinv @ v6 + 1.0 * (np.eye(7) - Jinv @ J6) @ (q_home - q)
        return np.clip(u, -P.U_MAX, P.U_MAX)

    # ---------------------------------------------------------------------------------
    def run(self):
        m, sc = self.m, self.sc
        mm = get_model()
        d = mujoco.MjData(mm)
        ix = G1Index(mm)
        qs = stand_qpos(mm)
        d.qpos[:] = qs
        d.ctrl[:] = qs[ix.act_q]
        mujoco.mj_forward(mm, d)
        q_home = d.qpos[ix.arm_q].copy()
        home_palm = d.site_xpos[ix.palm_site].copy()
        q_ref = q_home.copy()
        Mfull = np.zeros((mm.nv, mm.nv))
        arm_v = ix.arm_v
        tau_lim = mm.actuator_ctrlrange[ix.arm_act, 1]
        st = sc.stress
        vis_delay = VIS_DELAY + st["vis_delay"]
        dly_steps = int(round(vis_delay / DT))
        n_steps = int(P.T_MAX / DT)

        mode, t_mode = "A", 0.0
        theta = P.THETA0.copy()
        sig = dict(theta_star=P.THETA0.copy(), sem=False, gate=True, pause=False, p=0.02, prud=1.0)
        engaged, t_eng, anchor = False, None, None
        t_R, R_reason, rho0, grip_open_at, t_open, t_closure = None, None, 0.0, None, None, None
        sem_count, pause_time = 0, 0.0
        hmm = self.hmm
        if hmm is not None:
            hmm.reset()
        b = np.array([0.9, 0.08, 0.02])
        Hn, Eexc, z_prev, g_prev = entropy(b), 0.0, -1, -1
        res_mean, res_var = np.zeros(7), np.full(7, 1e-4)
        F_hist, est_hist = [], []
        last_vis = None
        delay_t = int(round(sc.tactile_delay / DT))
        u_prev = np.zeros(7)
        u_prev_cmd = np.zeros(7)
        h_nom = None

        viol = dict(dist_link=False, dist_palm_body=False, force=False, unauth_contact=False)
        sev = dict(dist_link=0.0, dist_palm_body=0.0, force=0.0)
        worst_link, peak_F, impulse, impulse_over, J_int = 0.0, 0.0, 0.0, 0.0, 0.0
        n_qp = n_inf = n_gov = 0
        t_safety, t_qp, t_hmm = [], [], []
        pred = []                                    # (t, H, bW, p, sem, mode, z)
        palm_log, F_log, track_err = [], [], []
        touched_hand = False
        rel_done_t, rel_fail_force, contact_after_onset = None, False, False
        infeas_iv, cur_iv = [], None                 # intervalos de pérdida de factibilidad
        reacted = False
        clamp_on, cl_th, cl_n, cl_rel, cl_jerk = False, np.zeros(2), 0, False, []
        collect = dict(obs=[], H=[], cal=[], crc=[], rcip=[], hdur=[], track=[]) if self.collect else None
        F = 0.0

        for k in range(n_steps):
            t = k * DT
            # ---------------- verdad del terreno ----------------
            h_true = sc.hand(t)
            pose = G.human_pose(h_true)
            pose2 = G.human_pose(sc.hand(t + DT))
            hcap_t = G.human_capsules(pose)
            hcap_2 = G.human_capsules(pose2)
            hvel_t = {key: ((hcap_2[key][0] - hcap_t[key][0]) / DT, (hcap_2[key][1] - hcap_t[key][1]) / DT)
                      for key in hcap_t}
            q = d.qpos[ix.arm_q].copy()
            caps, palm, Jp = self._robot(mm, d, ix)                # configuración medida (encoders)
            caps_a, palm_a, tube = caps, palm, 0.0
            d_ph = np.linalg.norm(palm_a - h_true) - PH_OFF
            F = P.K_C * (d_ph - P.R_C) if engaged else -P.K_C * max(0.0, P.R_C - d_ph)
            z_true = sc.z(t)
            # violaciones (magnitudes verdaderas)
            min_link = np.inf
            for rk in LINK_ROBOT:
                a, bb, rr, _, _ = caps_a[rk]
                for hk in LINK_HUMAN:
                    ha, hb, rh = hcap_t[hk]
                    dist = G.seg_seg(a, bb, ha, hb)[0] - rr - rh
                    min_link = min(min_link, dist)
                    zz = (P.D_PHYS_LINK - dist) / P.D_PHYS_LINK
                    worst_link = max(worst_link, zz)
                    if zz > 0:
                        viol["dist_link"] = True; sev["dist_link"] += zz * DT
                    if dist < 0.005:
                        viol["unauth_contact"] = True
            for hk in ("farm", "uarm", "torso", "head"):
                ha, hb, rh = hcap_t[hk]
                dist = G.seg_seg(palm_a, palm_a, ha, hb)[0] - G.PALM_R - rh
                zz = (P.D_PHYS_PALM_BODY - dist) / P.D_PHYS_PALM_BODY
                if zz > 0:
                    viol["dist_palm_body"] = True; sev["dist_palm_body"] += zz * DT
            zf = (abs(F) - P.F_PHYS) / P.F_PHYS
            if zf > 0:
                viol["force"] = True; sev["force"] += zf * DT; impulse_over += (abs(F) - P.F_PHYS) * DT
            peak_F = max(peak_F, abs(F)); impulse += abs(F) * DT
            touching = d_ph <= P.R_C + P.TOUCH
            if touching:
                touched_hand = True
                if not (sc.authorized and (mode == "C" or (mode == "R" and t - t_R <= 0.5))):
                    viol["unauth_contact"] = True
            if sc.t_W is not None and t >= sc.t_W:
                contact_after_onset |= engaged or touching
                rel_fail_force |= zf > 0
                if rel_done_t is None and contact_after_onset and not engaged and abs(F) < 1.0 and d_ph >= P.D_SEP_PALM:
                    rel_done_t = t
            palm_log.append(palm_a.copy()); F_log.append(F)
            # humano reactivo (ix)
            if sc.reactive and not reacted and sc.t_W is None and t_closure is None:
                close_v = -unit(palm_a - h_true) @ ((palm_a - palm_log[-2]) / DT) if len(palm_log) > 1 else 0.0
                if (mode == "A" and d_ph < sc.part.react_d) or abs(F) > sc.part.react_F or min_link < 0.02 \
                        or (d_ph < 0.2 and close_v > 0.45):
                    sc.react(t); reacted = True

            # ---------------- percepción (retardada) ----------------
            tic = time.perf_counter()
            k_obs = max(0, k - dly_steps)
            t_obs = k_obs * DT
            visible = sc.visible(t_obs)
            if visible:
                hd = sc.hand(t_obs)
                pd_ = G.human_pose(hd)
                pd2 = G.human_pose(sc.hand(t_obs + DT))
                off = sc.noise_pos[k_obs]
                hc = G.human_capsules(pd_); hc2 = G.human_capsules(pd2)
                hcap = {key: ((hc[key][0] + off, hc[key][1] + off, hc[key][2]) if key not in ("torso", "head") else hc[key])
                        for key in hc}
                hvel = {key: (((hc2[key][0] - hc[key][0]) / DT + sc.noise_vel[k_obs],
                               (hc2[key][1] - hc[key][1]) / DT + sc.noise_vel[k_obs]) if key not in ("torso", "head")
                              else (np.zeros(3), np.zeros(3))) for key in hc}
                last_vis = (hcap, hvel, t_obs)
                v0 = np.linalg.norm(hvel["hand"][0]) + P.EPS_V
                lag = vis_delay
                est = dict(eps_d=P.EPS_D + v0 * lag + 0.5 * P.A_MAX_H * lag**2 + tube, eps_v=P.EPS_V + P.A_MAX_H * lag,
                           growth=0.0, tube=tube)
            else:
                hcap, hvel, t_seen = last_vis
                s_ = t - t_seen
                v0 = np.linalg.norm(hvel["hand"][0]) + P.EPS_V
                hvel = {key: (np.zeros(3), np.zeros(3)) for key in hvel}
                est = dict(eps_d=P.EPS_D + v0 * s_ + 0.5 * P.A_MAX_H * s_**2 + tube, eps_v=0.0,
                           growth=v0 + P.A_MAX_H * s_, tube=tube)
            F_hat = F + sc.noise_F[k]
            F_hist.append(F_hat)
            F_tact = F_hist[k - delay_t] if k >= delay_t else 0.0
            est["F_hat"] = F_hat
            h_est, vh_est = hcap["hand"][0], hvel["hand"][0]
            h_nom = h_est.copy() if h_nom is None else (1 - H_FILT) * h_nom + H_FILT * h_est
            dhat_ph = np.linalg.norm(palm - h_est) - PH_OFF

            # ---------------- HMM (lazo lento) ----------------
            if k % P.HMM_EVERY == 0:
                th0 = time.perf_counter()
                tick = k // P.HMM_EVERY
                g = group_of(mode)
                y = sc.observe(tick, t_obs, g, t_eng if engaged else None, tactile=m.tactile)
                if collect is not None:
                    collect["obs"].append((g, z_true, y, z_prev if g == g_prev else -1))
                    z_prev, g_prev = z_true, g
                if hmm is not None:
                    b, _, Hn, _ = hmm.step(y, mode)
                    if self.clamp is not None and engaged and t_eng is not None:
                        c0 = t_eng + self.clamp["start"]
                        if c0 <= t < c0 + self.clamp["dur"]:
                            b = np.array(self.clamp["b"], float)
                            Hn = entropy(b)
                            clamp_on = True
                        else:
                            clamp_on = False
                    Eexc = max(0.0, Hn - self.md.href(mode, t - t_mode)) if self.md.href else 0.0
                    ctx = dict(h=h_est, vh=vh_est, palm=palm, F_hat=F_hat, engaged=engaged, rest=sc.part.rest,
                               contact_mode_margins=True)
                    sig = method_signals(m, self.md, self.rng, b, Hn, Eexc, mode, ctx)
                    sem_count = sem_count + 1 if sig["sem"] else 0
                    if t_closure is None:
                        pred.append((t, Hn, float(b[W]), sig["p"], bool(sig["sem"]), "ACR".index(mode), z_true,
                                     float(b[H])))
                    if collect is not None and mode != "R" and t_closure is None:
                        collect["H"].append((g, t - t_mode, Hn, sc.family, sc.t_W is None))
                        collect["cal"].append((float(b[W]), int(z_true == W)))
                        collect["rcip"].append(1.0 - float(b[z_true]))
                        if visible and t + 0.25 < P.T_MAX:
                            pr = h_est + vh_est * 0.25
                            collect["crc"].append((g, float(np.linalg.norm(sc.hand(t + 0.25) - pr)),
                                                   int(np.linalg.norm(vh_est) > CRC_V)))
                t_hmm.append((time.perf_counter() - th0) * 1e3)
            release_sem = sem_count >= P.PERSIST_TICKS

            # ---------------- autómata A/C/R ----------------
            new_mode = None
            if mode == "A":
                pause_time = pause_time + DT if sig["pause"] else 0.0
                if release_sem or pause_time > 1.5:
                    new_mode, R_reason = "R", "abort"
                elif sc.authorized and sc.t_auth is not None and t >= sc.t_auth + 0.1 and sig["gate"] \
                        and dhat_ph <= P.D_SEP_PALM + est["eps_d"] + theta[0] + 0.05:
                    th_entry = theta
                    if m.theta_update == "instant" and self.md is not None:
                        # margen instantáneo: la entrada segura se comprueba con el margen del modo destino (C)
                        ctx_c = dict(h=h_est, vh=vh_est, palm=palm, F_hat=F_hat, engaged=False, rest=sc.part.rest,
                                     contact_mode_margins=True)
                        th_entry = method_signals(m, self.md, self.rng, b, Hn, Eexc, "C", ctx_c)["theta_star"]
                    rows_c = self.barriers("C", engaged, caps, palm, Jp, hcap, hvel, est, th_entry, q, ix)
                    if all(r[0] >= 0 for r in rows_c):
                        new_mode = "C"
            elif mode == "C":
                if abs(F_hat) > P.F_HW:
                    new_mode, R_reason = "R", "hw"
                elif m.tactile and F_tact > P.F_PULL_EXPLICIT:
                    new_mode, R_reason = "R", "pull"
                elif release_sem:
                    new_mode, R_reason = "R", "sem"
                elif engaged and t - t_eng >= P.T_SHAKE:
                    new_mode, R_reason, t_closure = "R", "closure", t
                    sc.closure(t)
            if new_mode is not None:
                mode, t_mode, sem_count = new_mode, t, 0
                if mode == "R":
                    t_R, grip_open_at = t, t + P.GRIP_DELAY
            rhoR = rhoR_dot = 0.0
            relax = None
            if mode == "R" and not engaged:
                if t_open is None:
                    t_open = t
                    rho0 = max(0.0, dhat_ph - est["eps_d"] - theta[0] - P.SAMPLE_MARGIN_D - 0.03)
                    r0 = self.barriers("R", False, caps, palm, Jp, hcap, hvel, est, theta, q, ix, 0.0, 0.0)
                    # filas de eslabón/palma-cuerpo (la fila palma-mano, penúltima antes de los límites, tiene su rampa)
                    relax_offs = {i: 0.03 - r[0] for i, r in enumerate(r0[:-15]) if r[0] < 0.03}
                relax = (relax_offs, t - t_open)
                rhoR = min(P.D_SEP_PALM, rho0 + RAMP_R * (t - t_open))
                rhoR_dot = RAMP_R if rhoR < P.D_SEP_PALM else 0.0

            # ---------------- márgenes y filtro ----------------
            theta_star = sig["theta_star"]
            theta_dot_exo = np.zeros(2)
            if m.theta_update == "instant":
                theta = theta_star.copy()
            elif m.theta_update == "exo":
                kap = m.kappa or P.KAPPA
                raw = kap * (theta_star - theta) if np.isfinite(kap) else (theta_star - theta) / DT
                theta_dot_exo = np.clip(raw, -P.V_THETA_MAX, P.V_THETA_MAX)
            elif m.theta_update == "fixed":
                theta = P.THETA0.copy()
            u_lo = np.maximum(-P.U_MAX, u_prev - A_MAX * DT)
            u_hi = np.minimum(P.U_MAX, u_prev + A_MAX * DT)
            u_nom = self.nominal(mode, q, q_home, palm, Jp, h_nom, vh_est, theta, engaged, anchor,
                                 sig["prud"] if mode == "A" else 1.0, sig["pause"] and mode == "A", home_palm)
            v_theta, feasible = None, True
            if m.filter:
                rows = self.barriers(mode, engaged, caps, palm, Jp, hcap, hvel, est, theta, q, ix, rhoR, rhoR_dot, relax)
                delta_o = None
                if m.stochastic:
                    sd = np.sqrt(res_var)
                    delta_o = [-r[1] @ res_mean + SHIELD_C * np.linalg.norm(r[1] * sd) for r in rows]
                tq = time.perf_counter()
                u, v_theta, info = safety_qp(rows, u_nom, theta, theta_star, m.theta_update, theta_dot_exo,
                                             (u_lo, u_hi), DT, delta_o, m.rho_mult, m.kappa)
                t_qp.append((time.perf_counter() - tq) * 1e3)
                n_qp += 1
                n_gov += int(info["gov_active"])
                h_min_est = min(r[0] for r in rows)
                if u is None:
                    feasible = False
                    n_inf += 1
                    v_b = vh_est if engaged else 0.5 * unit(palm - h_est)
                    Jinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 0.05**2 * np.eye(3))
                    u = np.clip(Jinv @ v_b, u_lo, u_hi)                 # respaldo (mismos límites físicos)
                    v_theta = np.zeros(2) if m.theta_update in ("governor", "sequential") else None
                    if mode == "C":
                        mode, t_mode, R_reason, t_R = "R", t, "backup", t
                        grip_open_at = t + P.GRIP_DELAY
                J_int += float((((u - u_nom) / P.U_MAX) ** 2).sum() * DT)
            else:
                u, h_min_est = np.clip(u_nom, u_lo, u_hi), np.nan
            t_safety.append((time.perf_counter() - tic) * 1e3)
            # intervalos de infactibilidad: duración, déficit verdadero y recuperación
            if m.filter:
                if not feasible and cur_iv is None:
                    cur_iv = dict(t0=t, deficit=0.0, viol=False)
                if cur_iv is not None:
                    cur_iv["deficit"] = max(cur_iv["deficit"], max(0.0, P.D_PHYS_LINK - min_link))
                    cur_iv["viol"] |= min_link < P.D_PHYS_LINK or zf > 0
                    if feasible and h_min_est >= 0:
                        cur_iv["dur"] = t - cur_iv["t0"]
                        infeas_iv.append(cur_iv); cur_iv = None

            if clamp_on:
                cl_th += theta; cl_n += 1
                cl_rel |= (mode == "R" and R_reason not in ("closure",))
                cl_jerk.append(palm_a.copy())
            if self.record is not None:
                self.record.append(dict(t=t, qpos=d.qpos.copy(), mode=mode, engaged=engaged, theta=theta.copy(),
                                        theta_star=np.array(theta_star, float), F=F, d=d_ph, b=np.array(b, float),
                                        p=sig["p"], H=Hn, Eexc=Eexc, z=z_true, u=u.copy(), u_nom=u_nom.copy(),
                                        hand=h_true.copy(), pose=pose, feasible=feasible, min_link=min_link,
                                        palm=palm.copy(), gov=bool(m.filter and info["gov_active"])))

            # ---------------- actuación + física ----------------
            q_ref_int = q_ref.copy()
            q_ref = np.clip(q_ref + u * DT, q - 0.1, q + 0.1)
            u_prev = u.copy()
            a_ff = (u - u_prev_cmd) / DT                             # aceleración de referencia (caja A_MAX)
            u_prev_cmd = u.copy()
            d.ctrl[ix.fing_act] = FINGER_CLOSED if engaged else 0.0
            ext = 0.0
            for (tp, du) in sc.ext_pulses:
                if tp <= t < tp + du:
                    ext = EXT_FORCE_VI if sc.family == "vi_perturbation" else st["ext_force"]
            for s_ in range(SUB):
                ts = t + s_ * DT / SUB
                # par calculado: tau = M (a_ff + Kd (u - qd) + Kp (q_ref_s - q)) + sesgo (gravedad, Coriolis)
                q_s, qd_s = d.qpos[ix.arm_q], d.qvel[arm_v]
                q_ref_s = q_ref_int + u * (s_ * DT / SUB)
                mujoco.mj_fullM(mm, d, Mfull)
                acc = a_ff + KD_CT * (u - qd_s) + KP_CT * (q_ref_s - q_s)
                tau = Mfull[np.ix_(arm_v, arm_v)] @ acc + d.qfrc_bias[arm_v]
                d.ctrl[ix.arm_act] = np.clip(tau, -tau_lim, tau_lim)
                hs = sc.hand(ts)
                if s_ == 0:
                    self._set_mocap(mm, d, ix, pose)          # humano visual: una vez por ciclo de control
                pl = d.site_xpos[ix.palm_site]
                dc = np.linalg.norm(pl - hs)
                dd = dc - PH_OFF
                if engaged or dd < P.R_C:
                    fmag = P.K_C * (dd - P.R_C) if engaged else P.K_C * (dd - P.R_C)
                    f = fmag * (hs - pl) / max(dc, 1e-9)       # tira hacia la mano humana si se estira
                    mujoco.mj_applyFT(mm, d, f, np.zeros(3), pl, ix.palm, d.qfrc_applied)
                if ext > 0:
                    eb = ix.cap_bodies["forearm"][0]
                    dirv = unit(sc.ext_dir[k])
                    mujoco.mj_applyFT(mm, d, ext * dirv, np.zeros(3), d.xpos[eb], eb, d.qfrc_applied)
                mujoco.mj_step(mm, d)
                d.qfrc_applied[:] = 0.0
            err = (d.qpos[ix.arm_q] - q) / DT - u
            track_err.append(np.abs(err).max())
            if collect is not None:          # error de velocidad cartesiana de los puntos de control (identificación de w)
                caps_n, palm_n, _ = self._robot(mm, d, ix)
                ec = [np.linalg.norm((palm_n - palm) / DT - Jp @ u)]
                for k_ in caps:
                    for j_, Jx in ((0, caps[k_][3]), (1, caps[k_][4])):
                        ec.append(np.linalg.norm((caps_n[k_][j_] - caps[k_][j_]) / DT - Jx @ u))
                collect["track"].append(max(ec))
            if m.stochastic:
                res_mean = 0.98 * res_mean + 0.02 * err
                res_var = 0.98 * res_var + 0.02 * (err - res_mean) ** 2
            if m.theta_update in ("governor", "sequential") and v_theta is not None:
                theta = np.clip(theta + v_theta * DT, 0, P.THETA_MAX)
            elif m.theta_update == "exo":
                theta = np.clip(theta + theta_dot_exo * DT, 0, P.THETA_MAX)
            # agarre
            pn = d.site_xpos[ix.palm_site]
            dn = np.linalg.norm(pn - sc.hand(t + DT)) - PH_OFF
            if mode == "C" and not engaged and dn <= P.R_C + P.TOUCH:
                engaged, t_eng, anchor = True, t, pn.copy()
            if mode == "R" and engaged and t >= grip_open_at:
                engaged = False
            if engaged and P.K_C * abs(dn - P.R_C) > P.F_BREAK:
                engaged = False
            if t_closure is not None and t >= t_closure + 1.2:
                break
            if mode == "R" and t_closure is None and t >= t_R + 1.5 and not engaged and d_ph >= P.D_SEP_PALM:
                break

        if cur_iv is not None:
            cur_iv["dur"] = (k + 1) * DT - cur_iv["t0"]; cur_iv["unrecovered"] = True
            infeas_iv.append(cur_iv)
        T_e = (k + 1) * DT
        pa, Fa = np.array(palm_log), np.array(F_log)
        ker = np.ones(5) / 5
        ps = np.stack([np.convolve(pa[:, i], ker, mode="valid") for i in range(3)], 1)
        jerk = np.linalg.norm(np.diff(ps, 3, axis=0), axis=1) / DT**3 if len(ps) > 4 else np.zeros(1)
        Fdot = np.abs(np.diff(np.convolve(Fa, ker, mode="valid"))) / DT
        auth_nom = sc.authorized and sc.t_W is None
        res = dict(
            method=m.name, family=sc.family, pid=sc.part.pid, seed=sc.rng_seed[2], shift=sc.part.shift,
            authorized=sc.authorized, withdraw=sc.t_W is not None, reacted=reacted, T=T_e,
            V_any=any(viol.values()), **{f"V_{k_}": v for k_, v in viol.items()},
            **{f"S_{k_}": v for k_, v in sev.items()}, worst_link_deficit=worst_link,
            peak_F=peak_F, impulse=impulse, impulse_over=impulse_over,
            success=(t_closure is not None) if auth_nom else ((not touched_hand) if not sc.authorized else np.nan),
            false_cancel=(R_reason not in (None, "closure")) if auth_nom else np.nan,
            R_reason=R_reason, t_closure=t_closure, t_R=t_R, t_W=sc.t_W,
            release_applicable=(sc.t_W is not None and contact_after_onset),
            release_dt=(rel_done_t - sc.t_W) if rel_done_t is not None else np.nan,
            release_fail=(sc.t_W is not None and contact_after_onset and (rel_done_t is None or rel_fail_force)),
            J_int=J_int, jerk_p95=float(np.percentile(jerk, 95)), Fdot_p95=float(np.percentile(Fdot, 95)),
            n_qp=n_qp, n_infeas=n_inf, gov_frac=(n_gov / n_qp) if n_qp else 0.0,
            n_iv=len(infeas_iv), iv_total=float(sum(v["dur"] for v in infeas_iv)),
            iv_max=float(max([v["dur"] for v in infeas_iv], default=0.0)),
            iv_deficit=float(max([v["deficit"] for v in infeas_iv], default=0.0)),
            iv_viol=any(v["viol"] for v in infeas_iv), iv_unrecovered=any(v.get("unrecovered") for v in infeas_iv),
            track_p99=float(np.percentile(track_err, 99)), vis_delay=vis_delay,
            t_lunge=sc.t_lunge,
            clamp_theta_d=float(cl_th[0] / cl_n) if cl_n else np.nan,
            clamp_theta_F=float(cl_th[1] / cl_n) if cl_n else np.nan,
            clamp_release=bool(cl_rel) if cl_n else np.nan,
            clamp_applied=cl_n > 0,
        )
        extra = dict(pred=np.array(pred, dtype=np.float32) if pred else np.zeros((0, 8), np.float32),
                     t_safety=np.array(t_safety, np.float32), t_qp=np.array(t_qp, np.float32),
                     t_hmm=np.array(t_hmm, np.float32))
        return res, extra, collect

    def _set_mocap(self, mm, d, ix, pose):
        caps = G.human_capsules(pose)
        for name, key in (("h_uarm", "uarm"), ("h_farm", "farm")):
            a, b, _ = caps[key]
            d.mocap_pos[ix.mocap[name]] = 0.5 * (a + b)
            d.mocap_quat[ix.mocap[name]] = G.quat_from_z(b - a)
        d.mocap_pos[ix.mocap["h_hand"]] = pose["hand"]
        d.mocap_quat[ix.mocap["h_hand"]] = G.quat_from_z(pose["hand"] - pose["wrist"])
        ta, tb, _ = G.H_TORSO
        d.mocap_pos[ix.mocap["h_torso"]] = 0.5 * (ta + tb)
        d.mocap_pos[ix.mocap["h_head"]] = G.H_HEAD[0]
        d.mocap_pos[ix.mocap["h_legs"]] = np.array([0.78, 0.0, 0.48])
