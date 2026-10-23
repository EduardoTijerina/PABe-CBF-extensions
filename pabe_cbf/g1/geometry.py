"""Geometría 3D: distancia segmento-segmento (cápsulas), humano cinemático y escena del saludo."""
import math

import numpy as np

from ..scenarios import Geometry

# --- escena (metros, marco del mundo; el G1 mira hacia +x) ---------------------------------
H_TORSO = (np.array([0.78, 0.0, 0.98]), np.array([0.78, 0.0, 1.32]), 0.15)   # cápsula torso
H_HEAD = (np.array([0.78, 0.0, 1.50]), 0.10)
H_SHOULDER = np.array([0.74, 0.17, 1.30])
H_LEN = (0.29, 0.26)            # brazo, antebrazo (hasta la muñeca)
HAND_LEN = 0.07                 # muñeca -> centro de la mano
R_HAND, R_FARM, R_UARM = 0.045, 0.040, 0.050
PALM_R = 0.03                   # radio efectivo de la palma del robot
ELBOW_POLE = np.array([0.15, 0.6, -1.0])
ROBOT_REF = np.array([0.0, -0.10, 1.0])   # referencia para la velocidad radial observada

G1_GEOM = Geometry(meet=[0.36, -0.07, 1.00], rest=[0.80, 0.30, 0.86], shoulder=ROBOT_REF,
                   lat=[0.0, 1.0, 0.0], shake=[0.0, 0.0, 1.0], lunge_target=[0.16, -0.24, 0.88], n_dof=7)


def _cl(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def seg_seg(p1, q1, p2, q2):
    """Puntos más cercanos entre los segmentos [p1,q1] y [p2,q2] (Ericson, RTCD 5.1.9).
    Devuelve (distancia, s, t, c1, c2)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s, t = 0.0, _cl(f / e)
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, _cl(-c / a)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = _cl((b * f - c * e) / den) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, _cl(-c / a)
            elif t > 1:
                t, s = 1.0, _cl((b - c) / a)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    dv = c1 - c2
    return math.sqrt(float(dv @ dv)), s, t, c1, c2


def cross3(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def human_pose(hand, off=None):
    """Brazo humano por IK de dos eslabones con polo del codo hacia abajo y afuera.
    off: desplazamiento del cuerpo humano respecto de su puesto (familias con caminata; Scenario.body)."""
    off = np.zeros(3) if off is None else off
    sh = H_SHOULDER + off
    v = sh - hand
    wrist = hand + HAND_LEN * v / (np.linalg.norm(v) + 1e-9)
    a, b = H_LEN
    d = wrist - sh
    D = np.clip(np.linalg.norm(d), abs(a - b) + 1e-3, a + b - 1e-3)
    u = d / (np.linalg.norm(d) + 1e-9)
    x = (a * a - b * b + D * D) / (2 * D)
    r = np.sqrt(max(a * a - x * x, 0.0))
    pv = ELBOW_POLE - (ELBOW_POLE @ u) * u
    pv = pv / (np.linalg.norm(pv) + 1e-9)
    elbow = sh + x * u + r * pv
    return dict(hand=hand, wrist=wrist, elbow=elbow, shoulder=sh, off=off)


def human_capsules(pose):
    """Cápsulas humanas (a, b, radio) usadas por las barreras."""
    o = pose["off"]
    return {"hand": (pose["hand"], pose["hand"], R_HAND),
            "farm": (pose["elbow"], pose["wrist"], R_FARM),
            "uarm": (pose["shoulder"], pose["elbow"], R_UARM),
            "torso": (H_TORSO[0] + o, H_TORSO[1] + o, H_TORSO[2]),
            "head": (H_HEAD[0] + o, H_HEAD[0] + o, H_HEAD[1])}


def quat_from_z(v):
    """Cuaternión (w,x,y,z) que lleva el eje z a la dirección v."""
    z = np.array([0.0, 0.0, 1.0])
    v = v / (np.linalg.norm(v) + 1e-12)
    c = z @ v
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    ax = cross3(z, v)
    q = np.array([1.0 + c, *ax])
    return q / np.linalg.norm(q)
