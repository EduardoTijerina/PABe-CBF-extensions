"""Seis familias de escenarios (Sec. 7.5) con humano de lazo abierto y semillas apareadas.

El humano sigue un guion independiente del método evaluado (mismas trayectorias por semilla),
excepto la retirada de cierre tras un apretón exitoso, que es reactiva.
"""
import numpy as np
from . import params as P

FAMILIES = ("i_stable", "ii_withdraw_pre", "iii_withdraw_contact",
            "iv_hesitation", "v_occlusion", "vi_perturbation")
# familias adicionales del banco G1: pérdida temporal de factibilidad, retirada ambigua, humano reactivo
FAMILIES_G1 = FAMILIES + ("vii_lunge", "viii_ambiguous_withdraw", "ix_reactive")
E, H, W = 0, 1, 2
MU2 = np.array([1.0, 0.0, -1.0])     # señal visual/proxémica de disposición
MU3 = np.array([1.0, 0.4, -0.6])     # señal táctil de agarre


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _bump(s):
    """0 -> 1 -> 0 suave en s in [0,1]."""
    s = np.clip(s, 0.0, 1.0)
    return np.sin(np.pi * s) ** 2


class Geometry:
    """Geometría de la escena (planar o 3D) usada por los guiones humanos."""

    def __init__(self, meet, rest, shoulder, lat, shake, lunge_target, n_dof=3):
        self.meet, self.rest, self.shoulder = np.asarray(meet, float), np.asarray(rest, float), np.asarray(shoulder, float)
        self.lat, self.shake = np.asarray(lat, float), np.asarray(shake, float)
        self.lunge_target = np.asarray(lunge_target, float)
        self.dim, self.n_dof = len(self.meet), n_dof


PLANAR = Geometry(P.MEET, P.HAND_REST, P.SHOULDER, [0.0, 1.0], [1.0, 0.0], [0.30, -0.25], n_dof=3)


class Participant:
    def __init__(self, pid, shift=False, geom=PLANAR):
        rng = np.random.default_rng([7919, pid])
        self.pid = pid
        self.geom = geom
        self.speed = rng.uniform(0.8, 1.25)
        self.f2_off = rng.normal(0.0, 0.25)
        self.meet = geom.meet + rng.uniform(-0.03, 0.03, geom.dim)
        self.rest = geom.rest + rng.uniform(-0.03, 0.03, geom.dim)
        self.react_d = rng.uniform(0.04, 0.08)     # (ix) distancia de incomodidad sin autorización
        self.react_F = rng.uniform(9.0, 13.0)      # (ix) fuerza de incomodidad
        self.shift = shift   # cambio de dominio: señales menos informativas y más ruidosas


class Scenario:
    def __init__(self, family, part, ep, stress=None):
        self.family = family
        self.part = part
        self.g = g = part.geom
        st = dict(noise_scale=1.0, vis_delay=0.0, obs_noise=1.0, ext_force=0.0, lunge_amp=0.12, tactile_delay=None)
        st.update(stress or {})
        self.stress = st
        fi = FAMILIES_G1.index(family)
        rng = np.random.default_rng([fi, part.pid, ep, 104729])
        rng2 = np.random.default_rng([fi, part.pid, ep, 271828])   # aleatoriedad exclusiva del banco G1
        self.rng_seed = (fi, part.pid, ep)
        rest, meet = part.rest, part.meet
        t_s = rng.uniform(0.2, 0.5)
        T_a = rng.uniform(0.9, 1.3) / part.speed
        t_arr = t_s + T_a
        self.knots = [(0.0, rest), (t_s, rest), (t_arr, meet)]
        self.authorized = True
        self.t_auth = t_s + 0.7 * T_a
        self.z_windows = []            # (t0, t1, z)
        self.t_W = None                # inicio (ground truth) de la retirada
        self.shake = [t_arr + 0.6, np.inf]
        self.bumps = []                # (t0, dur, vector)
        self.ext_pulses = []           # (t0, dur) empujones externos (G1)
        self.t_lunge = None
        self.occlusion = None
        self.tactile_delay = 0.0
        self.w_max = P.W_MAX
        self.act_delay = 0
        self.ambiguous = False
        self.reactive = False
        withdraw_contact = False

        if family == "ii_withdraw_pre":
            self.authorized = False
            self.t_auth = None
            self.t_W = t_s + rng.uniform(0.45, 0.8) * T_a
            self.shake = [np.inf, np.inf]
        elif family == "iii_withdraw_contact":
            withdraw_contact = True
        elif family == "iv_hesitation":
            t_h0 = t_s + 0.4 * T_a
            L_h = rng.uniform(0.6, 1.0)
            p_mid = rest + 0.4 * (meet - rest)
            back = (rest - meet) / np.linalg.norm(rest - meet)
            t_arr = t_h0 + L_h + 0.6 * T_a
            self.knots = [(0.0, rest), (t_s, rest), (t_h0, p_mid), (t_h0 + 0.5 * L_h, p_mid + 0.03 * back),
                          (t_h0 + L_h, p_mid), (t_arr, meet)]
            self.z_windows.append((t_h0 - 0.1, t_h0 + L_h, H))
            self.t_auth = t_h0 + L_h + 0.4 * 0.6 * T_a
            self.shake = [t_arr + 0.6, np.inf]
            t_c0 = t_arr + rng.uniform(0.9, 1.3)
            self.bumps.append((t_c0, 0.6, 0.012 * back))
            self.z_windows.append((t_c0 - 0.05, t_c0 + 0.6, H))
        elif family == "v_occlusion":
            withdraw_contact = rng.random() < 0.5
            t_occ = t_arr - 0.2 + rng.uniform(0.0, 1.0)
            self.occlusion = (t_occ, t_occ + 0.6)
            self.tactile_delay = 0.25
        elif family == "vi_perturbation":
            withdraw_contact = rng.random() < 0.5
            self.w_max = 0.10
            self.act_delay = 2
            self.bumps.append((t_arr + rng.uniform(0.8, 1.4), 0.2, 0.03 * g.lat * rng.choice([-1, 1])))
            self.ext_pulses = [(t_arr + rng2.uniform(0.3, 2.0), 0.15)]  # empujón externo sobre el antebrazo
        elif family == "vii_lunge":
            # embestida breve hacia el antebrazo del robot durante la aproximación (z = E): fuerza al filtro
            # a un escape que puede exceder la capacidad del robot -> pérdida temporal de factibilidad
            t_l = t_s + rng2.uniform(0.55, 0.8) * T_a
            p_l = self._base(t_l)
            self.bumps.append((t_l, 0.30, st["lunge_amp"] * (g.lunge_target - p_l) / np.linalg.norm(g.lunge_target - p_l)))
            self.t_lunge = t_l
        elif family == "viii_ambiguous_withdraw":
            withdraw_contact = True
            self.ambiguous = True       # señales de retirada débiles y retirada lenta: alta entropía con W real
        elif family == "ix_reactive":
            self.reactive = True        # se retira si el robot invade su espacio o tira demasiado fuerte

        if withdraw_contact:
            self.t_W = t_arr + rng.uniform(1.0, 1.6)

        if self.t_W is not None:
            react = rng.uniform(0.15, 0.25) if not self.ambiguous else rng.uniform(0.3, 0.5)
            t_r = self.t_W + react
            p_r = self._base(t_r)
            dur = rng.uniform(0.6, 0.9) if not self.ambiguous else rng.uniform(1.6, 2.2)
            self.knots = [k for k in self.knots if k[0] < t_r] + [(t_r, p_r), (t_r + dur, rest)]
            self.shake[1] = min(self.shake[1], t_r)
            self.bumps = [b for b in self.bumps if b[0] + b[1] < self.t_W]

        self.retract_at = None
        n_steps = int(P.T_MAX / P.DT) + 5
        dim = g.dim
        ns = st["noise_scale"]
        self.noise_pos = ns * rng.uniform(-P.NOISE_POS, P.NOISE_POS, (n_steps, dim))
        self.noise_vel = ns * rng.uniform(-0.5 * P.EPS_V, 0.5 * P.EPS_V, (n_steps, dim))
        self.noise_F = rng.uniform(-P.NOISE_F, P.NOISE_F, n_steps)
        self.w = rng.uniform(-self.w_max, self.w_max, (n_steps, g.n_dof))
        if st["tactile_delay"] is not None:
            self.tactile_delay = st["tactile_delay"]
        self.noise_obs = rng.standard_normal((n_steps // P.HMM_EVERY + 5, 3))
        self.ext_dir = rng2.normal(0, 1, (n_steps, dim))
        if st["ext_force"] > 0 and family != "vi_perturbation":
            # factor de sensibilidad: tres empujones de 0,2 s sobre el antebrazo en instantes aleatorios
            self.ext_pulses = self.ext_pulses + [(float(tp), 0.2) for tp in rng2.uniform(0.5, 5.0, 3)]

    # ----- trayectoria de la mano -------------------------------------------------
    def _base(self, t, knots=None):
        knots = knots or self.knots
        if t <= knots[0][0]:
            return knots[0][1].copy()
        for (t0, p0), (t1, p1) in zip(knots[:-1], knots[1:]):
            if t <= t1:
                return p0 + _minjerk((t - t0) / max(t1 - t0, 1e-6)) * (p1 - p0)
        return knots[-1][1].copy()

    def hand(self, t):
        if self.retract_at is not None and t >= self.retract_at[0]:
            p = self._base(t, self.retract_at[1])
        else:
            p = self._base(t)
            t0, t1 = self.shake
            if t0 <= t < t1:
                env = min(1.0, (t - t0) / 0.2, max(0.0, (t1 - t) / 0.2))
                p = p + self.g.shake * 0.008 * env * np.sin(2 * np.pi * 1.5 * (t - t0))
        for (tb, dur, vec) in self.bumps:
            if tb <= t <= tb + dur:
                p = p + vec * _bump((t - tb) / dur)
        return p

    def hand_vel(self, t):
        e = 1e-3
        return (self.hand(t + e) - self.hand(t - e)) / (2 * e)

    def closure(self, t):
        """El robot completó el apretón: la persona retira la mano tras 0.3 s (no es retirada W)."""
        if self.retract_at is None:
            p = self.hand(t)
            self.retract_at = (t, [(t, p), (t + 0.3, p), (t + 1.1, self.part.rest)])

    def react(self, t):
        """(ix) Retirada provocada por el robot: z = W desde t y retirada a reposo."""
        if self.t_W is None and self.retract_at is None:
            self.t_W = t
            p = self.hand(t)
            self.knots = [(t, p), (t + 0.15, p), (t + 0.85, self.part.rest)]
            self.shake = [np.inf, np.inf]
            self.bumps = []

    def z(self, t):
        if self.t_W is not None and t >= self.t_W:
            return W
        for t0, t1, zz in self.z_windows:
            if t0 <= t < t1:
                return zz
        return E

    def visible(self, t):
        return self.occlusion is None or not (self.occlusion[0] <= t < self.occlusion[1])

    # ----- observaciones multimodales -------------------------------------------
    def observe(self, tick, t, group, engaged_since, tactile=True):
        """y_t = (velocidad radial visual, disposición proxémica, señal táctil); NaN = ausente."""
        n = self.noise_obs[tick]
        z = self.z(t)
        y = np.full(3, np.nan)
        scale, sdk = (0.6, 1.3) if self.part.shift else (1.0, 1.0)
        sdk = sdk * self.stress["obs_noise"]
        mu2, mu3 = MU2[z], MU3[z]
        if self.ambiguous and z == W:
            mu2, mu3 = -0.35, 0.2                 # señales de retirada débiles (persona indecisa)
        if self.visible(t):
            h = self.hand(t)
            r = (h - self.g.shoulder) / np.linalg.norm(h - self.g.shoulder)
            y[0] = -self.hand_vel(t) @ r + 0.06 * sdk * n[0]
            sd2 = (0.6 if group == 1 else 0.5) * sdk
            y[1] = mu2 * scale + self.part.f2_off + sd2 * n[1]
        if tactile and group == 1 and engaged_since is not None and t - engaged_since >= self.tactile_delay:
            if t - engaged_since < 0.3 + self.tactile_delay:
                y[2] = 0.3 + 1.0 * n[2]          # transitorio ambiguo al establecer contacto
            else:
                y[2] = mu3 * scale + 0.5 * sdk * n[2]
        return y
