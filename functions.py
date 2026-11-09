"""Function families, verified ground truth, and interval evaluation.

Every function is a sympy expression with exactly representable rational
coefficients; numpy versions are used for data generation and training, and
mpmath interval versions are used for the rigorous certificates."""
import numpy as np
import sympy as sp
from mpmath import iv
from scipy.optimize import brentq

X = sp.Symbol("x", real=True)
A_DOM, B_DOM = -10.0, 10.0
N_SAMPLES = 101
GRID = np.linspace(A_DOM, B_DOM, N_SAMPLES)
STAB_THR = 0.95
iv.dps = 25


def q(v, digits=4):
    """Exact rational with `digits` decimals."""
    return sp.Rational(int(round(v * 10 ** digits)), 10 ** digits)


# ----------------------------------------------------------------------------
# training families
# ----------------------------------------------------------------------------
def poly_expr(rng, deg, base=2.0, damp=True):
    e = 0
    for i in range(deg + 1):
        c = rng.uniform(-base, base)
        if damp and i >= 2:
            c *= 10.0 ** (-(i - 1))
        e += q(c, 6) * X ** i
    return e


def trig(rng):
    A, w, phi = rng.uniform(0.5, 3), rng.uniform(0.5, 2), rng.uniform(0, 2 * np.pi)
    B, C = rng.uniform(-0.5, 0.5), rng.uniform(-5, 5)
    return q(A) * sp.sin(q(w) * X + q(phi)) + q(B) * X + q(C)


def expo(rng):
    A = rng.uniform(0.1, 1) * rng.choice([-1, 1])
    B, C, D = rng.uniform(-1, 1), rng.uniform(-2, 2), rng.uniform(-5, 5)
    return q(A) * sp.exp(q(B) * X) + q(C) * X + q(D)


def poly(rng):
    return poly_expr(rng, int(rng.integers(2, 6)))


def rational(rng):
    num = poly_expr(rng, int(rng.integers(0, 4)), base=3.0)
    u = rng.random()
    if u < 0.3:
        num += trig(rng)
    elif u < 0.5:
        num += q(rng.uniform(-1, 1)) * sp.exp(q(rng.uniform(-0.3, 0.3)) * X)
    tt = np.linspace(-11, 11, 2001)
    for _ in range(200):
        den = poly_expr(rng, int(rng.integers(1, 4)), base=1.0)
        fd = sp.lambdify(X, den, "numpy")
        vals = np.broadcast_to(fd(tt), tt.shape)
        if np.min(np.abs(vals)) >= 0.5:
            return num / den
    return None


def composite(rng):
    k = int(rng.integers(0, 4))
    B, C = q(rng.uniform(-0.8, 0.8)), q(rng.uniform(-5, 5))
    A = q(rng.uniform(0.5, 4))
    if k == 0:
        return A * sp.sin(poly_expr(rng, 2, base=0.6)) + B * X + C
    if k == 1:
        m, s = q(rng.uniform(-8, 8)), q(rng.uniform(0.5, 10))
        return A * sp.exp(-(X - m) ** 2 / s) + B * X + C
    if k == 2:
        w, phi, s = q(rng.uniform(0.5, 3)), q(rng.uniform(0, 6.28)), q(rng.uniform(5, 60))
        return A * sp.sin(w * X + phi) * sp.exp(-X ** 2 / s) + B * X + C
    S = sp.sin(X)
    return sum(q(rng.uniform(-3, 3)) * S ** i for i in range(4)) + B * X + C


FAMILIES = {"trigonometric": trig, "exponential": expo, "polynomial": poly,
            "rational": rational, "composite": composite}


# ----------------------------------------------------------------------------
# out-of-distribution families
# ----------------------------------------------------------------------------
def pwl(rng):
    kn = np.sort(rng.uniform(-8, 8, 2))
    sl = rng.uniform(-2.5, 2.5, 3)
    c0 = rng.uniform(-6, 6)
    k1, k2 = q(kn[0]), q(kn[1])
    s1, s2, s3 = q(sl[0]), q(sl[1]), q(sl[2])
    y1 = q(c0) + s1 * k1
    y2 = y1 + s2 * (k2 - k1)
    return sp.Piecewise((q(c0) + s1 * X, X < k1), (y1 + s2 * (X - k1), X < k2), (y2 + s3 * (X - k2), True))


def kink(rng):
    return q(rng.uniform(-2, 2)) * sp.Abs(X - q(rng.uniform(-6, 6))) + q(rng.uniform(-1, 1)) * X + q(rng.uniform(-5, 5))


def sigmoid_mix(rng):
    e = sum(q(rng.uniform(-4, 4)) * sp.tanh(q(rng.uniform(0.3, 3)) * (X - q(rng.uniform(-8, 8)))) for _ in range(3))
    return e + q(rng.uniform(-0.5, 0.5)) * X + q(rng.uniform(-3, 3))


def conj(expr_nat, p, qq):
    """Affine conjugation of a map on [p, qq] to [-10, 10] (preserves multipliers)."""
    p, qq = sp.nsimplify(p), sp.nsimplify(qq)
    hinv = p + (X + 10) * (qq - p) / 20
    y = expr_nat.subs(X, hinv)
    return -10 + 20 * (y - p) / (qq - p)


def quad_map(rng):
    a = q(rng.uniform(0.5, 4))
    c = q(rng.uniform(-0.1, 0.1))
    return conj(a * X * (1 - X) + c, -0.25, 1.25)


OOD = {"Piecewise-linear": pwl, "Kink maps": kink, "Sigmoid mixtures": sigmoid_mix, "Quadratic maps": quad_map}


# classical dynamical systems (natural interval, then conjugated)
def logistic(rng):
    return conj(q(rng.uniform(0.5, 4)) * X * (1 - X), 0, 1)


def ricker(rng):
    return conj(X * sp.exp(q(rng.uniform(0.5, 3)) * (1 - X)), 0, 4)


def bev_holt(rng):
    R = q(rng.uniform(0.5, 5))
    return conj(R * X / (1 + (R - 1) * X), 0, 2)


def henon(rng):
    return conj(1 - q(rng.uniform(0.1, 1.4)) * X ** 2, -1.5, 1.5)


def sine_map(rng):
    return conj(q(rng.uniform(0.3, 1.0)) * sp.sin(sp.pi * X), 0, 1)


DYNSYS = {"Logistic map": logistic, "Beverton--Holt": bev_holt, "H\\'enon slice": henon,
          "Sine map": sine_map, "Ricker map": ricker}


# applications
def market(rng):
    k, s0, s1 = q(rng.uniform(0.1, 1)), q(rng.uniform(-1, 1)), q(rng.uniform(0.2, 1.5))
    d0, d1, d2 = q(rng.uniform(-1, 2)), q(rng.uniform(0.2, 1.5)), q(rng.uniform(-3, 3))
    return X + k * ((d0 - d1 * X - d2 * sp.tanh(X)) - (s0 + s1 * X))


def control(rng):
    a, Kp, r = q(rng.uniform(0, 2)), q(rng.uniform(0.1, 2.5)), q(rng.uniform(-5, 5))
    return X + a * sp.sin(X) + Kp * (r - X)


def chem(rng):
    A, k1, k2, dt = q(rng.uniform(0.5, 2)), q(rng.uniform(0.5, 2)), q(rng.uniform(0.1, 1.5)), q(rng.uniform(0.1, 1.5))
    return conj(X + dt * (k1 * X * (A - X) - k2 * X), -0.25, 2.5)


APPS = {"Economics": (market, 200), "Control": (control, 150), "Chemical kinetics": (chem, 100)}


# ----------------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------------
def numpy_funcs(expr):
    f = sp.lambdify(X, expr, "numpy")
    df = sp.lambdify(X, sp.diff(expr, X), "numpy")
    F = lambda t: np.broadcast_to(np.asarray(f(t), float), np.shape(t)).copy()
    DF = lambda t: np.broadcast_to(np.asarray(df(t), float), np.shape(t)).copy()
    return F, DF


def scalar_func(expr):
    try:
        fm = sp.lambdify(X, expr, "math")
        fm(0.5)
        return lambda t: float(fm(t))
    except Exception:
        return None


def find_fixed_points(f, df, ngrid=4001, fs=None):
    fs = (lambda s: f(np.array([s]))[0]) if fs is None else fs
    t = np.linspace(A_DOM, B_DOM, ngrid)
    g = f(t) - t
    roots = []
    for i in range(ngrid - 1):
        if g[i] == 0:
            roots.append(t[i])
        elif g[i] * g[i + 1] < 0:
            try:
                r = brentq(lambda s: fs(s) - s, t[i], t[i + 1], xtol=1e-15, rtol=1e-15, maxiter=200)
            except ValueError:
                continue
            roots.append(r)
    # tangential roots: local minima of |g| that are tiny
    ag = np.abs(g)
    for i in range(1, ngrid - 1):
        if ag[i] < ag[i - 1] and ag[i] <= ag[i + 1] and ag[i] < 1e-3 and g[i - 1] * g[i + 1] > 0:
            r = t[i]
            for _ in range(50):
                d = df(np.array([r]))[0] - 1
                if d == 0:
                    break
                r -= (f(np.array([r]))[0] - r) / d
            if abs(f(np.array([r]))[0] - r) < 1e-10 and A_DOM <= r <= B_DOM:
                roots.append(r)
    out = []
    for r in sorted(roots):
        # Newton polish
        for _ in range(3):
            d = df(np.array([r]))[0] - 1
            if not np.isfinite(d) or d == 0:
                break
            rn = r - (f(np.array([r]))[0] - r) / d
            if abs(rn - r) < 1e-6:
                r = rn
        if abs(f(np.array([r]))[0] - r) < 1e-10 and A_DOM <= r <= B_DOM:
            if not out or abs(r - out[-1]) > 1e-7:
                out.append(float(r))
    return out


def basins(f, roots, grid=GRID, iters=300):
    x = grid.copy()
    with np.errstate(all="ignore"):
        for _ in range(iters):
            x = f(x)
            x = np.where(np.isfinite(x) & (np.abs(x) < 1e8), x, np.nan)
    lab = -np.ones(len(grid), int)
    for j, r in enumerate(roots):
        lab[np.abs(x - r) < 1e-6] = j
    return lab


def analyze(expr, max_fp=10, reject=True):
    try:
        f, df = numpy_funcs(expr)
        with np.errstate(all="ignore"):
            y = f(GRID)
            if not np.all(np.isfinite(y)) or (reject and np.max(np.abs(y)) > 1e6):
                return None
            roots = find_fixed_points(f, df, fs=scalar_func(expr))
    except Exception:
        return None
    if reject and (len(roots) == 0 or len(roots) > max_fp):
        return None
    if len(roots) == 0:
        return None
    with np.errstate(all="ignore"):
        d = df(np.array(roots))
    if not np.all(np.isfinite(d)):
        return None
    return dict(samples=y.astype(np.float64), roots=np.array(roots), deriv=d,
                stable=(np.abs(d) < STAB_THR).astype(int), basin=basins(f, roots))


# ----------------------------------------------------------------------------
# interval evaluation (for certificates)
# ----------------------------------------------------------------------------
def _tanh_iv(Z):
    Z = iv.mpf(Z)
    t = lambda a: 1 - 2 / (iv.exp(2 * iv.mpf(a)) + 1)
    return iv.mpf([t(Z.a).a, t(Z.b).b])


_NS = {"sin": iv.sin, "cos": iv.cos, "exp": iv.exp, "tanh": _tanh_iv, "mpf": iv.mpf, "pi": iv.pi}


def interval_funcs(expr):
    g = sp.lambdify(X, expr - X, modules=[_NS, "mpmath"])
    dg = sp.lambdify(X, sp.diff(expr, X), modules=[_NS, "mpmath"])
    return g, dg
