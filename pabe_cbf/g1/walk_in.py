"""Visualización de caminatas: el G1 y la persona caminan uno hacia el otro y luego se saludan.

Dos usos:
  * familias con caminata (Scenario.walk, p. ej. x_walk_hesitation): el episodio físico ya incluye la
    aproximación en el marco del robot; aquí sólo se pasa al marco del mundo (la base del robot avanza) y se
    animan las piernas;
  * cualquier otra familia: se antepone una caminata cinemática (prólogo) al episodio registrado.

En ambos casos la marcha es una animación cinemática: la simulación tiene base fija y no hay controlador de
locomoción; el brazo derecho y el saludo son los del episodio físico.
"""
import mujoco
import numpy as np

from ..scenarios import Walk, _knots_interp
from . import geometry as G
from .episode import G1Episode
from .model import G1Index, HUMAN_GEOMS, build, stand_qpos

T_SETTLE = 0.4       # prólogo: pausa al llegar, antes del saludo (s)
STRIDE_R = 0.8       # ciclo de marcha completo (dos pasos) del G1 (m)
STRIDE_H = 1.2       # ídem, persona (pasos de 0,6 m)
V_GAIT = 0.5         # velocidad a la que la oscilación de marcha alcanza su amplitud completa (m/s)
PELVIS_Z = 0.793
H_HIP = np.array([0.78, 0.09, 0.88])   # cadera derecha humana en su puesto (la izquierda es simétrica en y)
H_LEG_LEN = 0.84
# prólogo: misma caminata del robot que en las familias con caminata; la persona camina sin dudar
PROLOGUE_H = [(0.0, Walk.D_R), (Walk.T_R, 0.0)]

_MODEL = None


def get_walk_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build(walk=True)
    return _MODEL


def _gait(knots, t, D0, stride):
    """Amplitud (velocidad normalizada) y fase (ligada a la distancia recorrida: el pie no resbala)."""
    e = 1e-3
    v = abs(_knots_interp(t + e, knots) - _knots_interp(t - e, knots)) / (2 * e)
    return min(1.0, v / V_GAIT), 2 * np.pi * (D0 - _knots_interp(t, knots)) / stride


class WalkHandshake:
    """Línea de tiempo renderizable de un episodio registrado (G1Episode(..., record=True))."""

    def __init__(self, record, sc, dt=0.01):
        self.rec, self.sc, self.dt = record, sc, dt
        self.mm = mm = get_walk_model()
        self.ix = G1Index(mm)
        self.qs = stand_qpos(mm)
        B = lambda n: mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pelvis = mm.body_mocapid[B("pelvis")]
        self.legs = (mm.body_mocapid[B("h_rleg")], mm.body_mocapid[B("h_lleg")])
        self.jq = lambda n: mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)]
        if sc.walk is not None:
            # marcha reactiva del robot: la registrada en el episodio (nodos con velocidad, cada 5 ciclos)
            ts = np.array([x["t"] for x in record]); xs = np.array([x["x_r"] for x in record])
            vs = np.gradient(xs, ts) if len(ts) > 1 else np.zeros_like(xs)
            idx = sorted(set(range(0, len(ts), 5)) | {len(ts) - 1})
            self.knots_r = [(float(ts[i]), float(xs[i]), float(vs[i])) for i in idx]
            self.knots_h, self.n_pro = sc.walk.knots_h, 0
        else:
            self.knots_r, self.knots_h = [(0.0, Walk.D_R), (Walk.T_R, 0.0)], PROLOGUE_H
            self.n_pro = int(round((Walk.T_R + T_SETTLE) / dt))
        self.t_pro = self.n_pro * dt
        self.n = self.n_pro + len(record)

    def time(self, i):
        return i * self.dt if i < self.n_pro else self.t_pro + self.rec[i - self.n_pro]["t"]

    def frame(self, i):
        """Registro del fotograma i: en el prólogo, el primer estado del episodio (persona en reposo)."""
        return self.rec[max(0, i - self.n_pro)]

    def apply(self, d, i):
        """Escribe qpos y mocap del fotograma i en d (marco del mundo) y llama a mj_forward."""
        mm, ix = self.mm, self.ix
        tw = self.time(i)                  # tiempo de caminata (en las familias con caminata, el del episodio)
        r = self.frame(i)
        x_r = _knots_interp(tw, self.knots_r)
        x_h = _knots_interp(tw, self.knots_h)
        env_r, ph_r = _gait(self.knots_r, tw, self.knots_r[0][1], STRIDE_R)
        env_h, ph_h = _gait(self.knots_h, tw, self.knots_h[0][1], STRIDE_H)
        # robot: el brazo derecho es el del episodio; piernas y brazo izquierdo, oscilación de marcha
        d.qpos[:] = r["qpos"]
        self._robot_gait(d.qpos, ph_r, env_r)
        d.mocap_pos[self.pelvis] = [-x_r, 0.0, PELVIS_Z - 0.012 * env_r * np.sin(ph_r) ** 2]
        # persona: pose del episodio (marco del robot) -> mundo; en el prólogo el registro no incluye la caminata
        G1Episode._set_mocap(None, mm, d, ix, r["pose"])
        body_x = x_h + x_r if self.sc.walk is None else 0.0
        shift = np.array([body_x - x_r, 0.0, -0.012 * env_h * np.sin(ph_h) ** 2])
        for name in HUMAN_GEOMS:
            d.mocap_pos[ix.mocap[name]] += shift
        hip_x = H_HIP[0] + x_h
        for k, sgn in ((0, 1.0), (1, -1.0)):
            a = 0.35 * env_h * np.sin(ph_h) * sgn
            hip = np.array([hip_x, sgn * H_HIP[1], H_HIP[2] + shift[2]])
            u = np.array([-np.sin(a), 0.0, -np.cos(a)])    # la persona mira hacia -x
            d.mocap_pos[self.legs[k]] = hip + 0.5 * H_LEG_LEN * u
            d.mocap_quat[self.legs[k]] = G.quat_from_z(u)
        mujoco.mj_forward(mm, d)

    def _robot_gait(self, q, ph, env):
        """Oscilación de marcha sobre la postura de pie (env = 0: sin cambios). No toca el brazo derecho."""
        jq = self.jq
        sn, cs = np.sin(ph), np.cos(ph)
        for side, sgn in (("right", 1.0), ("left", -1.0)):
            swing = max(0.0, sgn * cs)                      # fase de vuelo de esa pierna
            q[jq(f"{side}_hip_pitch_joint")] -= 0.35 * sgn * sn * env
            q[jq(f"{side}_knee_joint")] += 0.6 * swing * env
            q[jq(f"{side}_ankle_pitch_joint")] -= 0.3 * swing * env
        q[jq("left_shoulder_pitch_joint")] -= 0.3 * sn * env  # brazo opuesto a la pierna
