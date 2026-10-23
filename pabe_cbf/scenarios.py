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
# familias con caminata de aproximación (sólo G1; fuera de FAMILIES_G1 para no alterar entrenamiento ni banco original)
FAMILIES_WALK = ("x_walk_hesitation",)
FAMILIES_ALL = FAMILIES_G1 + FAMILIES_WALK
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


def _knots_interp(t, knots):
    """Interpolación entre nodos (t_k, x_k) o (t_k, x_k, v_k): Hermite quíntico con aceleración nula en los
    nodos (v_k = 0 por defecto, que es el perfil de mínimo jerk). Constante fuera del intervalo."""
    if t <= knots[0][0]:
        return knots[0][1]
    for k0, k1 in zip(knots[:-1], knots[1:]):
        if t <= k1[0]:
            h = max(k1[0] - k0[0], 1e-6)
            s = (t - k0[0]) / h
            v0, v1 = (k0[2] if len(k0) > 2 else 0.0), (k1[2] if len(k1) > 2 else 0.0)
            s3, s4, s5 = s**3, s**4, s**5
            return (k0[1] + _minjerk(s) * (k1[1] - k0[1])
                    + h * (v0 * (s - 6 * s3 + 8 * s4 - 3 * s5) + v1 * (-4 * s3 + 7 * s4 - 3 * s5)))
    return knots[-1][1]


class Walk:
    """Caminata de aproximación (familia x): robot y persona caminan uno hacia el otro desde D_R + D_h.

    En el marco del robot (base fija en la simulación) la persona entera se desplaza: su separación extra
    respecto del puesto del saludo es lo que le falta caminar a ella más lo que le falta caminar al robot.
    La marcha del robot es reactiva: G1Episode fija su velocidad en cada ciclo (step) a partir de las señales
    del método (conciencia proxémica) y aquí se integra con límites de velocidad y aceleración y frenado en
    su puesto. La forma de la caminata de la persona es una variante aleatoria por episodio (VARIANTS), de
    lazo abierto; al llegar ambos, el saludo sigue la lógica habitual.

      stop_back      se detiene, medio paso atrás, sigue                       H en la pausa
      slowdown       frena al 25-40 % de la velocidad sin detenerse            H en el tramo lento
      double_pause   dos paradas breves                                        H en cada una
      false_alarm    parada breve sin paso atrás (se distrae)                  E (cancelar es falsa cancelación)
      abandon        se detiene, duda y se aleja sin llegar                    H y luego W; no hay saludo
      arrival_delay  llega con normalidad y tarda en ofrecer la mano           H de pie en su puesto
    """
    D_R, V_R, A_R = 1.6, 0.5, 0.6     # el robot camina 1,6 m; velocidad (m/s) y aceleración (m/s²) máximas
    T_R = D_R / V_R + V_R / A_R       # duración sin detenerse (perfil trapezoidal), ~4,0 s
    PROX_GAP = 0.25                   # la señal proxémica de aproximación existe mientras la persona está a más
                                      # de esto de su puesto (torso a torso > ~1 m); luego la sustituyen las del brazo
    VARIANTS = ("stop_back", "slowdown", "double_pause", "false_alarm", "abandon", "arrival_delay")

    def __init__(self, rng, speed):
        self.variant = v = self.VARIANTS[rng.integers(len(self.VARIANTS))]
        D = rng.uniform(1.4, 1.8)
        T = 4.0 / speed                           # caminata sin pausas (pico ~0,7 m/s a velocidad 1)
        tw = lambda dx: T * dx / D                # duración de un tramo de parada a parada
        self.z_windows = []                       # (t0, t1, z) durante la caminata
        self.t_leave = None                       # abandon: inicio de la retirada (W)
        delay = 0.0                               # arrival_delay: espera antes de ofrecer la mano
        if v in ("stop_back", "false_alarm", "abandon"):
            f = rng.uniform(0.35, 0.6)            # fracción del camino antes de detenerse
            L = rng.uniform(0.8, 1.5) if v != "false_alarm" else rng.uniform(0.5, 1.0)
            x1, t1 = (1 - f) * D, tw(f * D)
            t2 = t1 + L
            if v == "stop_back":
                back = rng.uniform(0.05, 0.12)    # medio paso atrás
                K = [(0.0, D), (t1, x1), (t1 + 0.5 * L, x1 + back), (t2, x1), (t2 + tw(x1) + 0.3, 0.0)]
            elif v == "false_alarm":
                K = [(0.0, D), (t1, x1), (t2, x1), (t2 + tw(x1), 0.0)]
            else:
                r = rng.uniform(0.6, 1.0)         # se aleja r metros, despacio
                K = [(0.0, D), (t1, x1), (t2, x1), (t2 + r / 0.25, x1 + r)]
                self.t_leave = t2
            if v != "false_alarm":
                self.z_windows.append((t1 - 0.1, t2, H))
        elif v == "slowdown":
            f, Ls, k = rng.uniform(0.25, 0.35), rng.uniform(0.8, 1.3), rng.uniform(0.25, 0.4)
            vc = 1.5 * D / T                      # velocidad de crucero
            xa, ta = (1 - f) * D, f * D / (0.6 * vc)
            dx = 0.5 * (1 + k) * vc * 0.5 * Ls    # distancia de cada mitad del tramo lento
            tm, tb = ta + 0.5 * Ls, ta + Ls
            xb = xa - 2 * dx
            K = [(0.0, D, 0.0), (ta, xa, -vc), (tm, xa - dx, -k * vc), (tb, xb, -vc), (tb + xb / (0.6 * vc), 0.0, 0.0)]
            self.z_windows.append((ta, tb, H))
        elif v == "double_pause":
            f1, f2 = rng.uniform(0.25, 0.4), rng.uniform(0.55, 0.75)
            L1, L2 = rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.8)
            x1, x2 = (1 - f1) * D, (1 - f2) * D
            t1 = tw(f1 * D)
            t2 = t1 + L1 + tw(x1 - x2)
            K = [(0.0, D), (t1, x1), (t1 + L1, x1), (t2, x2), (t2 + L2, x2), (t2 + L2 + tw(x2), 0.0)]
            self.z_windows += [(t1 - 0.1, t1 + L1, H), (t2 - 0.1, t2 + L2, H)]
        else:                                     # arrival_delay
            K = [(0.0, D), (T, 0.0)]
            delay = rng.uniform(0.8, 1.6)
        self.knots_h = K
        self.t_arrive = max(K[-1][0], self.T_R)   # fin de ambas caminatas (robot sin detenerse)
        self.t_start = self.t_arrive + delay      # a partir de aquí, guion habitual del saludo
        if delay:
            self.z_windows.append((self.t_arrive, self.t_start, H))
        self.dt = P.DT
        self.xr = [self.D_R]                      # posición del robot (lo que le falta) en k·DT
        self.vr = 0.0                             # velocidad actual del robot (m/s, hacia su puesto)

    def human(self, t):
        return _knots_interp(t, self.knots_h)

    def human_speed(self, t):
        """Velocidad propia de la persona hacia su puesto (positiva al acercarse)."""
        e = 1e-3
        return -(self.human(t + e) - self.human(t - e)) / (2 * e)

    def step(self, v_des, dt):
        """Un ciclo de la marcha del robot hacia v_des, con aceleración acotada y frenado para parar en su puesto."""
        x = self.xr[-1]
        v = float(np.clip(v_des, self.vr - self.A_R * dt, self.vr + self.A_R * dt))
        v = max(0.0, min(v, self.V_R, np.sqrt(2 * self.A_R * x)))
        self.xr.append(max(0.0, x - v * dt))
        self.vr = v if self.xr[-1] > 0 else 0.0

    def robot(self, t):
        """Lo que le falta caminar al robot en t: historial integrado; más allá, extrapolación a velocidad actual."""
        k = t / self.dt
        n = len(self.xr) - 1
        if k <= 0:
            return self.xr[0]
        if k <= n:
            i = int(k)
            j = min(i + 1, n)
            return self.xr[i] + (k - i) * (self.xr[j] - self.xr[i])
        return max(0.0, self.xr[-1] - self.vr * (k - n) * self.dt)

    def robot_rate(self, t):
        """d/dt de robot(t) (negativa mientras el robot avanza)."""
        e = 1e-3
        return (self.robot(t + e) - self.robot(t - e)) / (2 * e)


class Scenario:
    def __init__(self, family, part, ep, stress=None):
        self.family = family
        self.part = part
        self.g = g = part.geom
        st = dict(noise_scale=1.0, vis_delay=0.0, obs_noise=1.0, ext_force=0.0, lunge_amp=0.12, tactile_delay=None)
        st.update(stress or {})
        self.stress = st
        fi = FAMILIES_ALL.index(family)
        rng = np.random.default_rng([fi, part.pid, ep, 104729])
        rng2 = np.random.default_rng([fi, part.pid, ep, 271828])   # aleatoriedad exclusiva del banco G1
        self.rng_seed = (fi, part.pid, ep)
        rest, meet = part.rest, part.meet
        self.walk = None
        t0 = 0.0                       # inicio del saludo (tras la caminata, si la hay)
        if family in FAMILIES_WALK:
            self.walk = Walk(np.random.default_rng([fi, part.pid, ep, 161803]), part.speed)
            t0 = self.walk.t_start
        # con caminata, margen para que el robot pueda esperar a la persona (marcha reactiva)
        self.t_max = P.T_MAX + t0 + (4.0 if self.walk is not None else 0.0)
        t_s = t0 + rng.uniform(0.2, 0.5)
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
            # G1 (3D): duda visible -- a mitad del alcance la mano se retira 15 cm hacia el cuerpo y vuelve a
            # ofrecerse (pausa 1,5x más larga), y en contacto un tirón de 2 cm (6 N con K_C, bajo el umbral de
            # tirón explícito de 8 N). Banco planar: amplitudes originales (3 cm y 1,2 cm).
            big = g.dim == 3
            t_h0 = t_s + 0.4 * T_a
            L_h = rng.uniform(0.6, 1.0) * (1.5 if big else 1.0)
            p_mid = rest + 0.4 * (meet - rest)
            back = (rest - meet) / np.linalg.norm(rest - meet)
            t_arr = t_h0 + L_h + 0.6 * T_a
            self.knots = [(0.0, rest), (t_s, rest), (t_h0, p_mid),
                          (t_h0 + 0.5 * L_h, p_mid + (0.15 if big else 0.03) * back), (t_h0 + L_h, p_mid), (t_arr, meet)]
            self.z_windows.append((t_h0 - 0.1, t_h0 + L_h, H))
            self.t_auth = t_h0 + L_h + 0.4 * 0.6 * T_a
            self.shake = [t_arr + 0.6, np.inf]
            t_c0 = t_arr + rng.uniform(0.9, 1.3)
            self.bumps.append((t_c0, 0.6, (0.02 if big else 0.012) * back))
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
        elif family == "x_walk_hesitation":
            # variante aleatoria de la caminata (Walk.VARIANTS); al llegar, saludo estable como en (i)
            self.z_windows += self.walk.z_windows
            if self.walk.t_leave is not None:      # abandon: se aleja sin llegar, no hay saludo autorizado
                self.authorized, self.t_auth = False, None
                self.t_W = self.walk.t_leave
                self.shake = [np.inf, np.inf]

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
        n_steps = int(self.t_max / P.DT) + 5
        dim = g.dim
        ns = st["noise_scale"]
        self.noise_pos = ns * rng.uniform(-P.NOISE_POS, P.NOISE_POS, (n_steps, dim))
        self.noise_vel = ns * rng.uniform(-0.5 * P.EPS_V, 0.5 * P.EPS_V, (n_steps, dim))
        self.noise_F = rng.uniform(-P.NOISE_F, P.NOISE_F, n_steps)
        self.w = rng.uniform(-self.w_max, self.w_max, (n_steps, g.n_dof))
        if st["tactile_delay"] is not None:
            self.tactile_delay = st["tactile_delay"]
        self.noise_obs = rng.standard_normal((n_steps // P.HMM_EVERY + 5, 3))
        # ruido del rasgo proxémico (y[3]) con su propio generador: no altera los demás sorteos
        self.noise_prox = np.random.default_rng([fi, part.pid, ep, 57721]).standard_normal(n_steps // P.HMM_EVERY + 5)
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
        return p if self.walk is None else p + self.body(t)

    def body(self, t):
        """Desplazamiento del cuerpo humano respecto de su puesto (marco del robot; el G1 mira hacia +x)."""
        off = np.zeros(self.g.dim)
        if self.walk is not None:
            off[0] = self.walk.human(t) + self.walk.robot(t)
        return off

    def hand_vel(self, t):
        e = 1e-3
        return (self.hand(t + e) - self.hand(t - e)) / (2 * e)

    def closure(self, t):
        """El robot completó el apretón: la persona retira la mano tras 0.3 s (no es retirada W)."""
        if self.retract_at is None:
            p = self.hand(t) - self.body(t)          # nodos en el marco de la persona
            self.retract_at = (t, [(t, p), (t + 0.3, p), (t + 1.1, self.part.rest)])

    def react(self, t):
        """(ix) Retirada provocada por el robot: z = W desde t y retirada a reposo."""
        if self.t_W is None and self.retract_at is None:
            self.t_W = t
            p = self.hand(t) - self.body(t)
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
        """y_t = (velocidad radial visual, disposición proxémica, señal táctil, aproximación del cuerpo); NaN = ausente.

        y[3] (sólo con caminata, persona visible y aún lejos de su puesto): velocidad propia con que la persona se
        acerca caminando, compensada la marcha del robot (odometría). Con caminata, y[0] también se compensa."""
        n = self.noise_obs[tick]
        z = self.z(t)
        y = np.full(4, np.nan)
        scale, sdk = (0.6, 1.3) if self.part.shift else (1.0, 1.0)
        sdk = sdk * self.stress["obs_noise"]
        mu2, mu3 = MU2[z], MU3[z]
        if self.ambiguous and z == W:
            mu2, mu3 = -0.35, 0.2                 # señales de retirada débiles (persona indecisa)
        if self.visible(t):
            h = self.hand(t)
            r = (h - self.g.shoulder) / np.linalg.norm(h - self.g.shoulder)
            hv = self.hand_vel(t)
            if self.walk is not None:
                hv = hv.copy()
                hv[0] -= self.walk.robot_rate(t)      # sin la parte debida a la marcha del propio robot
            y[0] = -hv @ r + 0.06 * sdk * n[0]
            sd2 = (0.6 if group == 1 else 0.5) * sdk
            y[1] = mu2 * scale + self.part.f2_off + sd2 * n[1]
            if self.walk is not None and self.body(t)[0] > Walk.PROX_GAP:
                y[3] = self.walk.human_speed(t) * scale + 0.08 * sdk * self.noise_prox[tick]
        if tactile and group == 1 and engaged_since is not None and t - engaged_since >= self.tactile_delay:
            if t - engaged_since < 0.3 + self.tactile_delay:
                y[2] = 0.3 + 1.0 * n[2]          # transitorio ambiguo al establecer contacto
            else:
                y[2] = mu3 * scale + 0.5 * sdk * n[2]
        return y
