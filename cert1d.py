"""Rigorous a posteriori certificates for fixed points of scalar maps.

All bounds are computed in interval arithmetic (mpmath.iv), so a successful
certificate is a proof about the exact function defined by the sympy expression.

  Banach   (Thm 1): sup|f'| on I_r = L < 1 and |g(mu)| <= (1-L) r
                    -> unique fixed point in I_r, stable, I_r inside its basin.
  Krawczyk (Thm 2): c = 1/(1-f'(mu)), kappa = sup|1 - c(1-f')| on I_r < 1 and
                    |c g(mu)| <= (1-kappa) r -> unique fixed point in I_r (any stability).
  Stability label : from the range of f' on the final enclosure I_rho.
"""
from mpmath import iv, mpf


def _hi(Z):
    return float(iv.mpf(Z).b)


def _lo(Z):
    return float(iv.mpf(Z).a)


def _absmax(Z):
    Z = iv.mpf(Z)
    return max(abs(float(Z.a)), abs(float(Z.b)))


def _absmin(Z):
    Z = iv.mpf(Z)
    a, b = float(Z.a), float(Z.b)
    return 0.0 if a <= 0 <= b else min(abs(a), abs(b))


class Certifier1D:
    def __init__(self, g_iv, dg_iv, dom=(-10.0, 10.0)):
        self.g, self.dg = g_iv, dg_iv      # g = f - id, dg = f'
        self.dom = dom
        self.n_point = 0                   # point (residual) evaluations
        self.n_int = 0                     # interval derivative evaluations

    def _gpt(self, x):
        self.n_point += 1
        return iv.mpf(self.g(iv.mpf(x)))

    def _dg(self, lo, hi):
        self.n_int += 1
        return iv.mpf(self.dg(iv.mpf([lo, hi])))

    def certify(self, mu, rmax=0.5, tries=6, slope=None):
        """Try Krawczyk (and Banach) on I_r for r = 2|c g|, 4|c g|, ... <= rmax.

        slope: float estimate of f'(mu) used for the preconditioner c."""
        mu = float(mu)
        res = dict(ok=False, banach=False, mu=mu)
        try:
            G = self._gpt(mu)
        except Exception:
            return res
        gabs = _absmax(G)
        if slope is None:
            try:
                slope = float(iv.mpf(self.dg(iv.mpf(mu))).mid)
                self.n_int += 1
            except Exception:
                return res
        if abs(1 - slope) < 1e-12:
            return res
        c = 1.0 / (1.0 - slope)
        cg = abs(c) * gabs * (1 + 1e-15)
        r = max(2 * cg, 1e-13)
        for _ in range(tries):
            if r > rmax:
                break
            lo, hi = mu - r, mu + r
            if lo < self.dom[0] - 1e-9 or hi > self.dom[1] + 1e-9:
                break
            try:
                D = self._dg(lo, hi)
            except Exception:
                break
            Kint = 1 - iv.mpf(c) * (1 - D)
            kap = _absmax(Kint)
            L = _absmax(D)
            if kap < 1 and cg <= (1 - kap) * r * (1 - 1e-15):
                rho = cg / (1 - kap) * (1 + 1e-15)
                res.update(ok=True, r=r, rho=rho, kappa=kap)
                if L < 1 and gabs <= (1 - L) * r * (1 - 1e-15):
                    res.update(banach=True, L=L, banach_rho=gabs / (1 - L) * (1 + 1e-15))
                try:
                    Dr = self._dg(mu - rho, mu + rho)
                    if _absmax(Dr) < 1:
                        res["stab"] = 1          # certified attracting
                    elif _absmin(Dr) > 1:
                        res["stab"] = 0          # certified repelling
                    else:
                        res["stab"] = -1         # undetermined
                except Exception:
                    res["stab"] = -1
                return res
            r *= 2
        return res
