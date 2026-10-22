"""Cinemática del brazo planar del robot y del brazo humano."""
import numpy as np
from . import params as P

ROBOT_POINTS = ("elbow", "fmid", "wrist", "palm")
HUMAN_POINTS = ("hand", "hwrist", "helbow", "torso")

# longitud efectiva de cada eslabón hasta el punto de interés
_POINT_LENS = {
    "elbow": (P.LINKS[0],),
    "fmid": (P.LINKS[0], 0.5 * P.LINKS[1]),
    "wrist": (P.LINKS[0], P.LINKS[1]),
    "palm": (P.LINKS[0], P.LINKS[1], P.LINKS[2]),
}


def fk(q):
    """Devuelve {punto: (posición (2,), jacobiano (2,3))}."""
    phi = np.cumsum(q)
    c, s = np.cos(phi), np.sin(phi)
    out = {}
    for name, lens in _POINT_LENS.items():
        p = P.SHOULDER.copy()
        J = np.zeros((2, 3))
        for j, l in enumerate(lens):
            p = p + l * np.array([c[j], s[j]])
            J[:, : j + 1] += (l * np.array([-s[j], c[j]]))[:, None]
        out[name] = (p, J)
    return out


def dls(J, v, lam=0.05):
    return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(2), v)


def ik_palm(target, q0=np.array([-1.0, 1.6, 0.2]), iters=200):
    q = q0.copy()
    for _ in range(iters):
        p, J = fk(q)["palm"]
        q = q + dls(J, target - p, 0.02)
    return q


Q_HOME = ik_palm(P.HOME_PALM)


def human_points(h):
    """Puntos humanos a partir de la mano: muñeca, codo (IK 2 eslabones, codo hacia fuera), torso."""
    v = P.H_SHOULDER - h
    dist = np.linalg.norm(v) + 1e-9
    hw = h + 0.08 * v / dist
    a, b = P.H_LINKS
    d = hw - P.H_SHOULDER
    D = np.clip(np.linalg.norm(d), abs(a - b) + 1e-3, a + b - 1e-3)
    base = np.arctan2(d[1], d[0])
    ang = np.arccos(np.clip((a * a + D * D - b * b) / (2 * a * D), -1, 1))
    cands = [P.H_SHOULDER + a * np.array([np.cos(base + s * ang), np.sin(base + s * ang)]) for s in (1, -1)]
    he = max(cands, key=lambda e: e[1])  # codo hacia +y (lejos de la línea media)
    return {"hand": h, "hwrist": hw, "helbow": he, "torso": P.TORSO}
