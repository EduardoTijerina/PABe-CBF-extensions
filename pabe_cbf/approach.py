"""Sistema de aproximación: marcha del robot con conciencia proxémica, independiente del sistema de saludo.

Actúa mientras robot y persona caminan uno hacia el otro (familias con caminata) y entrega el control al sistema
de saludo (PABe-CBF del brazo) cuando ambos llegan a su puesto (traspaso). Tiene su propia creencia: un HMM
E/H/W sobre las señales corporales y_a = (disposición proxémica y[1], velocidad propia de aproximación y[3]),
entrenado sólo con la fase de caminata, y su propia política de marcha. Políticas comparadas (banco A):

  P0  ciega: velocidad nominal, nunca aborta (lo que hacen B0/B1 sin creencia)
  P1  MAP: avanza sólo si el estado más probable es E; aborta si es W
  P2  entropía (como B3): r = b_W + 0,4 H; aborta si r > tau; velocidad V (1 - H)
  P3  PABe: aborta con p_W⁺ calibrada > tau; prudencia por entropía excedente (sobre H_ref); sin confianza
      (puerta cerrada, p_W⁺ >= 0,5) avanza despacio y no entra en el espacio personal

Abortar exige PERSIST_TICKS ticks seguidos, como la liberación semántica del saludo.
"""
from dataclasses import dataclass

import numpy as np

from . import params as P
from .belief import ModeHMM, entropy
from .scenarios import E, H, W, Walk

FEATS = [1, 3]                 # índices de y usados por el sistema de aproximación
D_PERSONAL = 1.2               # espacio personal (Hall), distancia torso-torso (m)
WALK_SLOW = 0.4                # fracción de la velocidad sin confianza en que la persona quiere el saludo
POLICIES = {"P0": "blind", "P1": "map", "P2": "entropy", "P3": "pabe"}
TAU_DEFAULT = {"P2": 0.6, "P3": P.TAU_REL_PABE}


@dataclass
class ApproachModels:
    A: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    prior: np.ndarray
    calib: object = None       # IsoUpperCalibrator de b_W (P3)
    href: float = None         # entropía de referencia en aproximaciones nominales (P3)


class ApproachSystem:
    def __init__(self, policy, models=None, tau=None):
        """policy: 'P0'..'P3', opcionalmente con umbral: 'P3@0.7'."""
        self.name = policy
        policy, _, t = policy.partition("@")
        tau = float(t) if t else tau
        self.policy = POLICIES[policy]
        self.md = models
        self.hmm = ModeHMM(models.A, models.mu, models.sd, models.prior) if models is not None else None
        if self.policy != "blind" and self.hmm is None:
            raise ValueError(f"la política {policy} necesita el HMM de aproximación (models.approach)")
        self.tau = tau if tau is not None else TAU_DEFAULT.get(policy, 1.0)
        self.reset()

    def reset(self):
        self.b = np.array([0.9, 0.08, 0.02])
        self.H = entropy(self.b)
        self.n_abort = 0
        self.sig = dict(abort=False, gate=True, prud=1.0, p=float(self.b[W]))
        if self.hmm is not None:
            self.hmm.reset()

    def update(self, y):
        """Un tick lento (33 Hz) con las señales observadas; devuelve las señales y_a usadas."""
        ya = np.asarray(y)[FEATS]
        if self.hmm is None:
            return ya
        self.b, _, self.H, _ = self.hmm.step(ya, "A")
        b, Hn, bW = self.b, self.H, float(self.b[W])
        if self.policy == "blind":
            sig = dict(abort=False, gate=True, prud=1.0, p=bW)
        elif self.policy == "map":
            z = int(np.argmax(b))
            sig = dict(abort=z == W, gate=z == E, prud=1.0 if z == E else 0.0, p=bW)
        elif self.policy == "entropy":
            r = bW + 0.4 * Hn
            sig = dict(abort=r > self.tau, gate=r < min(0.5, self.tau), prud=max(0.0, 1.0 - Hn), p=bW)
        else:
            p = self.md.calib.upper_of(bW)
            exc = max(0.0, Hn - self.md.href)
            sig = dict(abort=p > self.tau, gate=p < min(P.TAU_GATE_PABE, self.tau),
                       prud=max(0.3, 1.0 - P.C_PRUDENCE * exc), p=p)
        self.n_abort = self.n_abort + 1 if sig["abort"] else 0
        sig["abort"] = self.n_abort >= P.PERSIST_TICKS
        self.sig = sig
        return ya

    def command(self, d_body, v_now):
        """Velocidad de marcha deseada (m/s) y si se aborta la aproximación. d_body: distancia torso-torso percibida."""
        s = self.sig
        if s["abort"]:
            return 0.0, True
        v = Walk.V_R * s["prud"]
        if self.policy == "pabe" and not s["gate"]:
            stop_d = v_now**2 / (2 * Walk.A_R)                      # distancia de frenado
            v = WALK_SLOW * v if d_body - stop_d > D_PERSONAL + 0.02 else 0.0
        return v, False


def fit_hmm(records):
    """records: (z, y_a, z_prev | -1) de la fase de caminata."""
    hmm = ModeHMM.fit([(0, z, y, zp) for z, y, zp in records])
    return ApproachModels(hmm.A, hmm.mu, hmm.sd, hmm.prior)
