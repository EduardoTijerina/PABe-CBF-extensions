"""HMM condicionado por modo (Ec. 2), entropía (3), sorpresa, entropía excedente (4)
y calibración superior p_W^+ (Sec. 3.2). También calibradores de los baselines B5 y B8."""
import numpy as np
from scipy.stats import beta
from . import params as P

NZ = 3
LOG3 = np.log(3.0)


def group_of(mode):
    return 1 if mode == "C" else 0


def entropy(b):
    b = np.clip(b, 1e-12, 1.0)
    return float(-(b * np.log(b)).sum() / LOG3)


class ModeHMM:
    """Transición A^(m) por grupo de modo (A/R vs C) y emisiones robustas t de Student (nu=4)
    por estado y rasgo, con posición = mediana y escala = 1.4826·MAD. La t evita que un estado
    multimodal (W = 'retirándose' + 'ya retirado en reposo') quede mal representado por una gaussiana."""
    NU = 4.0

    def __init__(self, A, mu, sd, prior):
        self.A, self.mu, self.sd, self.prior = A, mu, sd, prior
        self.reset()

    @classmethod
    def fit(cls, records):
        """records: lista de (grupo, z, y(NF), z_prev | -1). Estimación supervisada con suavizado.
        Un rasgo sin muestras (siempre NaN) queda con escala 10 y nunca entra en la verosimilitud."""
        NF = len(records[0][2])
        A = np.ones((2, NZ, NZ)) * 0.5 + np.eye(NZ)[None] * 20.0
        vals = [[[[] for _ in range(NF)] for _ in range(NZ)] for _ in range(2)]
        for g, z, y, zp in records:
            if zp >= 0:
                A[g, zp, z] += 1
            for f in range(NF):
                if not np.isnan(y[f]):
                    vals[g][z][f].append(y[f])
        A /= A.sum(2, keepdims=True)
        mu = np.zeros((2, NZ, NF)); sd = np.full((2, NZ, NF), 10.0)
        for g in range(2):
            for z in range(NZ):
                for f in range(NF):
                    v = np.asarray(vals[g][z][f])
                    if len(v) >= 5:
                        mu[g, z, f] = np.median(v)
                        sd[g, z, f] = 1.4826 * np.median(np.abs(v - mu[g, z, f])) + 0.03
        prior = np.array([0.90, 0.08, 0.02])
        return cls(A, mu, sd, prior)

    def reset(self):
        self.b = self.prior.copy()

    def step(self, y, mode):
        g = group_of(mode)
        bbar = self.A[g].T @ self.b
        m = ~np.isnan(y)
        if m.any():
            r = (y[m] - self.mu[g][:, m]) / self.sd[g][:, m]
            ll = (-(self.NU + 1) / 2 * np.log1p(r**2 / self.NU)).sum(1) - np.log(self.sd[g][:, m]).sum(1)
            lik = np.exp(ll - ll.max())
        else:
            lik = np.ones(NZ)
        b = lik * bbar
        b = b / b.sum()
        b = (1 - 1e-4) * b + 1e-4 / NZ           # regularización numérica
        S = float((b * np.log(b / np.clip(bbar, 1e-12, 1))).sum())
        self.b = b
        return b, bbar, entropy(b), S


class HRef:
    """Entropía de transición esperada H_ref(m, tau) estimada en trazas nominales (cuantil q)."""
    BIN = 0.1
    NB = 40

    def __init__(self, table):
        self.table = table

    @classmethod
    def fit(cls, samples, q=0.8):
        """samples: lista de (grupo, tau, H)."""
        table = np.zeros((2, cls.NB))
        for g in (0, 1):
            for k in range(cls.NB):
                vals = [H for gg, tau, H in samples if gg == g and min(int(tau / cls.BIN), cls.NB - 1) == k]
                table[g, k] = np.quantile(vals, q) if len(vals) >= 10 else np.nan
            # rellenar huecos con el valor más cercano
            row = table[g]
            idx = np.where(~np.isnan(row))[0]
            for k in range(cls.NB):
                if np.isnan(row[k]):
                    row[k] = row[idx[np.argmin(abs(idx - k))]] if len(idx) else 0.5
        return cls(table)

    def __call__(self, mode, tau):
        return float(self.table[group_of(mode), min(int(tau / self.BIN), self.NB - 1)])


class UpperCalibrator:
    """Binning de b_W con cota superior de Clopper-Pearson al 95 % por bin (monótona).
    Bins con pocas muestras -> cota conservadora 1.0 (Sec. 3.2: sin calibración válida, ser conservador)."""

    def __init__(self, edges, upper, mean):
        self.edges, self.upper, self.mean = edges, upper, mean

    @classmethod
    def fit(cls, p, y, nbins=20, min_n=15, conf=0.95):
        p, y = np.asarray(p), np.asarray(y)
        edges = np.linspace(0, 1, nbins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
        upper = np.ones(nbins); mean = np.zeros(nbins)
        for k in range(nbins):
            yk = y[idx == k]
            nk, sk = len(yk), int(yk.sum())
            mean[k] = (sk + 0.5) / (nk + 1.0) if nk else edges[k] + 0.05
            if nk >= min_n:
                upper[k] = 1.0 if sk == nk else float(beta.ppf(conf, sk + 1, nk - sk))
        upper = np.maximum.accumulate(upper)
        mean = np.maximum.accumulate(mean)
        return cls(edges, upper, mean)

    def _bin(self, p):
        return min(max(int(np.digitize(p, self.edges)) - 1, 0), len(self.upper) - 1)

    def upper_of(self, p):
        return max(p, self.upper[self._bin(p)])

    def mean_of(self, p):
        return float(self.mean[self._bin(p)])


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


class IsoUpperCalibrator:
    """v2: calibración isotónica (PAV) que preserva el orden de b_W, con cota superior de
    Clopper-Pearson por bloque usando n efectivo = nº de episodios distintos del bloque
    (los ticks de un episodio están autocorrelados). La curva superior se interpola
    linealmente entre centros de bloque: continua, monótona y >= b_W."""

    def __init__(self, x, upper, mean):
        self.x, self.upper_c, self.mean_c = x, upper, mean

    @staticmethod
    def _pav(y, w):
        blocks = []                     # [suma_wy, suma_w, inicio, fin]
        for i, (yi, wi) in enumerate(zip(y, w)):
            blocks.append([yi * wi, wi, i, i])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                b = blocks.pop()
                blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][3] = b[3]
        return blocks

    @classmethod
    def fit(cls, p, y, ep, min_eff=20, conf=0.95, q=1e-3):
        p, y, ep = np.asarray(p, float), np.asarray(y, float), np.asarray(ep)
        # 1) agrupar valores casi iguales (evita escaleras artificiales por empates en el piso numérico)
        key = np.round(p / q).astype(int)
        uk, inv = np.unique(key, return_inverse=True)
        G = len(uk)
        n = np.bincount(inv, minlength=G).astype(float)
        sy = np.bincount(inv, weights=y, minlength=G)
        sp = np.bincount(inv, weights=p, minlength=G)
        eps_g = [set() for _ in range(G)]
        for g, e in zip(inv, ep):
            eps_g[g].add(e)
        # 2) PAV ponderado sobre los grupos
        blocks = []                      # [sy, n, sp, set_ep]
        for g in range(G):
            blocks.append([sy[g], n[g], sp[g], set(eps_g[g])])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                b = blocks.pop()
                for k in range(3):
                    blocks[-1][k] += b[k]
                blocks[-1][3] |= b[3]
        # 3) fusionar hasta >= min_eff episodios distintos por bloque
        merged = []
        for b in blocks:
            if merged and len(merged[-1][3]) < min_eff:
                for k in range(3):
                    merged[-1][k] += b[k]
                merged[-1][3] |= b[3]
            else:
                merged.append([b[0], b[1], b[2], set(b[3])])
        if len(merged) > 1 and len(merged[-1][3]) < min_eff:
            b = merged.pop()
            for k in range(3):
                merged[-1][k] += b[k]
            merged[-1][3] |= b[3]
        xs, up, mu = [], [], []
        for sy_, n_, sp_, es in merged:
            rate_ = sy_ / n_
            n_eff = len(es)
            k_eff = rate_ * n_eff
            hi = 1.0 if rate_ >= 1 else float(beta.ppf(conf, k_eff + 1, n_eff - k_eff))
            xs.append(sp_ / n_); up.append(hi); mu.append(rate_)
        up = np.maximum.accumulate(np.array(up)); mu = np.maximum.accumulate(np.array(mu))
        return cls(np.array(xs), up, mu)

    def upper_of(self, p):
        return float(max(p, np.interp(p, self.x, self.upper_c)))

    def mean_of(self, p):
        return float(np.interp(p, self.x, self.mean_c))


class ModeHSMM(ModeHMM):
    """A6: HSMM aproximado por expansión de estados. H se divide en k subestados en cadena (duración
    tipo Erlang/binomial negativa ajustada a la media y varianza de las duraciones de H observadas);
    E y W conservan duración geométrica. Emisiones idénticas al HMM; la creencia se agrega a (E,H,W)."""

    def __init__(self, base, k, A_exp):
        self.mu, self.sd, self.prior3 = base.mu, base.sd, base.prior
        self.k, self.A_exp = k, A_exp                 # A_exp: (2, k+2, k+2)
        self.reset()

    @classmethod
    def from_hmm(cls, base, h_durations_ticks):
        d = np.asarray(h_durations_ticks, float)
        mean = max(d.mean(), 2.0) if len(d) else 10.0
        var = d.var() if len(d) > 1 else mean
        k = int(np.clip(round(mean**2 / max(var + mean, 1e-6)), 1, 6))
        s = max(0.0, 1.0 - k / mean)                 # permanencia en cada subestado
        n = k + 2
        A_exp = np.zeros((2, n, n))
        for g in (0, 1):
            A = base.A[g]
            A_exp[g, 0, 0], A_exp[g, 0, 1], A_exp[g, 0, n - 1] = A[0, 0], A[0, 1], A[0, 2]
            for i in range(1, k + 1):
                A_exp[g, i, i] = s
                if i < k:
                    A_exp[g, i, i + 1] = 1 - s
                else:
                    out = A[1, 0] + A[1, 2]
                    A_exp[g, i, 0] = (1 - s) * (A[1, 0] / out if out > 0 else 1.0)
                    A_exp[g, i, n - 1] = (1 - s) * (A[1, 2] / out if out > 0 else 0.0)
            A_exp[g, n - 1, :] = 0.0
            A_exp[g, n - 1, 0], A_exp[g, n - 1, 1], A_exp[g, n - 1, n - 1] = A[2, 0], A[2, 1], A[2, 2]
            A_exp[g] /= A_exp[g].sum(1, keepdims=True)
        return cls(base, k, A_exp)

    def _agg(self, x):
        return np.array([x[0], x[1:-1].sum(), x[-1]])

    def reset(self):
        n = self.k + 2
        x = np.zeros(n)
        x[0], x[1], x[-1] = self.prior3[0], self.prior3[1], self.prior3[2]
        self.x = x
        self.b = self._agg(x)

    def step(self, y, mode):
        g = group_of(mode)
        xbar = self.A_exp[g].T @ self.x
        m = ~np.isnan(y)
        if m.any():
            r = (y[m] - self.mu[g][:, m]) / self.sd[g][:, m]
            ll = (-(self.NU + 1) / 2 * np.log1p(r**2 / self.NU)).sum(1) - np.log(self.sd[g][:, m]).sum(1)
            lik3 = np.exp(ll - ll.max())
        else:
            lik3 = np.ones(NZ)
        lik = np.concatenate([[lik3[0]], np.full(self.k, lik3[1]), [lik3[2]]])
        x = lik * xbar
        x = x / x.sum()
        x = (1 - 1e-4) * x + 1e-4 / len(x)
        self.x = x
        b, bbar = self._agg(x), self._agg(xbar)
        S = float((b * np.log(b / np.clip(bbar, 1e-12, 1))).sum())
        self.b = b
        return b, bbar, entropy(b), S
