# utils_copulas.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, chi2, t
from scipy import integrate, optimize

ArrayF = NDArray[np.floating]

# ---------- Exercise 2 functions ---------
def _validate_params(a: float, b: float, c: float) -> None:
    if not (a < c < b):
        raise ValueError(f"Require a < c < b, got a={a}, c={c}, b={b}.")


def tri_pdf(x: ArrayF, a: float, b: float, c: float) -> ArrayF:
    """
    PDF of Triangular(a,b,c).

    Parameters
    ----------
    x : array-like
        Evaluation points.
    a, b, c : float
        Parameters with a < c < b.

    Returns
    -------
    pdf : np.ndarray
        PDF values at x.
    """
    _validate_params(a, b, c)
    x = np.asarray(x, dtype=float)

    pdf = np.zeros_like(x, dtype=float)

    left = (x >= a) & (x <= c)
    right = (x >= c) & (x <= b)

    pdf[left] = 2.0 * (x[left] - a) / ((b - a) * (c - a))
    pdf[right] = 2.0 * (b - x[right]) / ((b - a) * (b - c))

    return pdf


def tri_ppf(u: ArrayF, a: float, b: float, c: float) -> ArrayF:
    """
    Quantile function (inverse CDF) of Triangular(a,b,c).
    """
    _validate_params(a, b, c)
    u = np.asarray(u, dtype=float)

    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("u must be in [0,1].")

    p = (c - a) / (b - a)

    x = np.empty_like(u, dtype=float)

    left = u <= p
    right = ~left

    x[left] = a + np.sqrt(u[left] * (b - a) * (c - a))
    x[right] = b - np.sqrt((1.0 - u[right]) * (b - a) * (b - c))

    return x


def tri_rvs(a: float, b: float, c: float, *, size: int, seed: int | None = None) -> ArrayF:
    """
    Draw samples from Triangular(a,b,c) via inverse transform sampling.
    """
    _validate_params(a, b, c)
    rng = np.random.default_rng(seed)
    u = rng.random(size)
    return tri_ppf(u, a, b, c)

try:
    from scipy.stats import norm
except ImportError as e:
    raise ImportError("Se requiere scipy: pip install scipy") from e

# ---------- Exercise 3 functions ----------

alpha, beta, gamma = 0.0642, 0.0049, 0.0296
r1_min, r1_max = -1, 2
r2_min, r2_max = -2, 2.5

def pdf_joint(r1, r2):
    if (r1_min <= r1 <= r1_max) and (r2_min <= r2 <= r2_max):
        return alpha + beta*r1 + gamma*r2
    return 0.0

def pdf_r1(r1):
    res, _ = integrate.quad(lambda r2: pdf_joint(r1, r2), r2_min, r2_max)
    return res

def cdf_r1(r1):
    res, _ = integrate.quad(pdf_r1, r1_min, r1)
    return res

# CORRECCIÓN: Inversa robusta ante errores de punto flotante
def inv_cdf_r1(u):
    if u <= 1e-6: return r1_min
    if u >= 1 - 1e-6: return r1_max
    # Verificamos los signos para evitar el error de brentq
    f_max = cdf_r1(r1_max) - u
    if f_max < 0: return r1_max # Si la integral no llega a u, devolvemos max
    return optimize.brentq(lambda r: cdf_r1(r) - u, r1_min, r1_max)

def pdf_r2(r2):
    res, _ = integrate.quad(lambda r1: pdf_joint(r1, r2), r1_min, r1_max)
    return res

def cdf_r2(r2):
    res, _ = integrate.quad(pdf_r2, r2_min, r2)
    return res

# CORRECCIÓN: Inversa robusta para R2
def inv_cdf_r2(u):
    if u <= 1e-6: return r2_min
    if u >= 1 - 1e-6: return r2_max
    f_max = cdf_r2(r2_max) - u
    if f_max < 0: return r2_max
    return optimize.brentq(lambda r: cdf_r2(r) - u, r2_min, r2_max)

def cond_exp_r1_given_r2(r2_val):
    denom = pdf_r2(r2_val)
    if denom < 1e-8: return 0 # Evitar división por cero
    num, _ = integrate.quad(lambda r1: r1 * pdf_joint(r1, r2_val), r1_min, r1_max)
    return num / denom

def cdf_cond_r1_given_r2(r1_target, r2_given):
    denom = pdf_r2(r2_given)
    if denom < 1e-8: return 0
    num, _ = integrate.quad(lambda t: pdf_joint(t, r2_given), r1_min, r1_target)
    val = num / denom
    return min(max(val, 0), 1) # Asegurar que esté entre 0 y 1

def conditional_copula_val(u1, u2):
    """
    Calcula C(u1|u2) = P(U1 <= u1 | U2 = u2).
    Matemáticamente equivale a F(r1 | r2) transformando los inputs al espacio real.
    """
    # 1. Transformar u2 -> r2 (espacio real original)
    r2 = inv_cdf_r2(u2)
    # 2. Transformar u1 -> r1 (espacio real original)
    r1 = inv_cdf_r1(u1)
    
    # 3. Calcular probabilidad condicional en el espacio real
    # Usamos la función cdf_cond_r1_given_r2 que definimos en el bloque anterior
    val = cdf_cond_r1_given_r2(r1, r2)
    return val

# ---------- Exercise 7 functions ----------
def gaussian_copula_uv(*, rho: float, size: int, seed: int | None = None) -> tuple[ArrayF, ArrayF]:
    if not (-1.0 < rho < 1.0):
        raise ValueError("Require -1 < rho < 1.")

    rng = np.random.default_rng(seed)

    # U2 ~ U(0,1)
    u2 = rng.random(size)
    z2 = norm.ppf(u2)

    # E ~ N(0,1) via inverse transform
    e = norm.ppf(rng.random(size))

    # Z1 | Z2=z2 ~ N(rho z2, 1-rho^2)
    z1 = rho * z2 + np.sqrt(1.0 - rho**2) * e
    u1 = norm.cdf(z1)

    return u1, u2

def check_normalization(alpha: float, beta: float, gamma: float, *, tol: float = 1e-10) -> None:
    mass = 13.5 * alpha + 6.75 * beta + 3.375 * gamma
    if not np.isclose(mass, 1.0, atol=tol, rtol=0.0):
        raise ValueError(f"No normalizado: 13.5α+6.75β+3.375γ = {mass} (debe ser 1).")


def f1_pdf(x: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = (x >= -1.0) & (x <= 2.0)
    out[mask] = 4.5 * (alpha + beta * x[mask]) + 1.125 * gamma
    return out


def f2_pdf(y: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    mask = (y >= -2.0) & (y <= 2.5)
    out[mask] = 3.0 * (alpha + gamma * y[mask]) + 1.5 * beta
    return out


def F1_cdf(x: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    A = 4.5 * alpha + 1.125 * gamma   # constant term in f1
    B = 4.5 * beta                    # slope term in f1

    out[x >= 2.0] = 1.0
    mid = (x > -1.0) & (x < 2.0)
    xm = x[mid]
    out[mid] = A * (xm + 1.0) + 0.5 * B * (xm**2 - 1.0)
    return out


def F2_cdf(y: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)

    A = 3.0 * alpha + 1.5 * beta      # constant term in f2
    B = 3.0 * gamma                   # slope term in f2

    out[y >= 2.5] = 1.0
    mid = (y > -2.0) & (y < 2.5)
    ym = y[mid]
    out[mid] = A * (ym + 2.0) + 0.5 * B * (ym**2 - 4.0)
    return out

def _ppf_linear_pdf_on_interval(u: ArrayF, *, L: float, U: float, A: float, B: float) -> ArrayF:
    u = np.asarray(u, dtype=float)
    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("u must be in [0,1].")

    # If B ~ 0 -> uniform on [L,U]
    if np.isclose(B, 0.0):
        # then CDF(x)=A(x-L), and A must be 1/(U-L)
        return L + u / A

    # Solve: A(x-L) + 0.5 B(x^2 - L^2) = u
    # => 0.5B x^2 + A x - (A L + 0.5B L^2 + u) = 0
    aq = 0.5 * B
    bq = A
    cq = -(A * L + 0.5 * B * L**2 + u)

    disc = bq**2 - 4.0 * aq * cq
    if np.any(disc < -1e-12):
        raise ValueError("Discriminante negativo: revisa parámetros (pdf puede no ser válida).")
    disc = np.maximum(disc, 0.0)

    sqrt_disc = np.sqrt(disc)
    x1 = (-bq + sqrt_disc) / (2.0 * aq)
    x2 = (-bq - sqrt_disc) / (2.0 * aq)

    # Elegir la raíz dentro de [L,U]
    in1 = (x1 >= L - 1e-12) & (x1 <= U + 1e-12)
    in2 = (x2 >= L - 1e-12) & (x2 <= U + 1e-12)

    x = np.where(in1 & ~in2, x1, np.where(in2 & ~in1, x2, x1))

    # Clamp final por robustez numérica
    return np.clip(x, L, U)


def F1_ppf(u: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    A = 4.5 * alpha + 1.125 * gamma
    B = 4.5 * beta
    return _ppf_linear_pdf_on_interval(u, L=-1.0, U=2.0, A=A, B=B)


def F2_ppf(u: ArrayF, alpha: float, beta: float, gamma: float) -> ArrayF:
    A = 3.0 * alpha + 1.5 * beta
    B = 3.0 * gamma
    return _ppf_linear_pdf_on_interval(u, L=-2.0, U=2.5, A=A, B=B)

def portfolio_returns(r1: ArrayF, r2: ArrayF, *, w1: float = 0.25, w2: float = 0.75) -> ArrayF:
    if not np.isclose(w1 + w2, 1.0):
        raise ValueError("Weights must sum to 1.")
    return w1 * r1 + w2 * r2


def var_left_tail(returns: ArrayF, alpha: float) -> float:
    q = float(np.quantile(returns, alpha))
    return -q


def normal_fit_pdf(x: ArrayF, grid: ArrayF) -> tuple[float, float, ArrayF]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    return mu, sigma, norm.pdf(grid, loc=mu, scale=sigma)

# ---------- Exercise 8 functions ----------
def gaussian_quantile_curve_r2(
    r1: np.ndarray,
    *,
    q: float,
    rho: float,
    alpha: float,
    beta: float,
    gamma: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Curva cuantil q bajo cópula Gaussiana:
      u1 = F1(r1)
      u2 = Phi( rho Phi^{-1}(u1) + sqrt(1-rho^2) Phi^{-1}(q) )
      r2 = F2^{-1}(u2)

    Usa las marginales derivadas del pdf (I): F1_cdf y F2_ppf (ya definidas en utils.py).
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1).")
    if not (-1.0 < rho < 1.0):
        raise ValueError("Require -1 < rho < 1.")

    # F1(r1) y estabilización para evitar ±inf en Phi^{-1}
    u1 = F1_cdf(r1, alpha, beta, gamma)
    u1 = np.clip(u1, eps, 1.0 - eps)

    z1 = norm.ppf(u1)
    zq = norm.ppf(q)

    u2 = norm.cdf(rho * z1 + np.sqrt(1.0 - rho**2) * zq)
    u2 = np.clip(u2, eps, 1.0 - eps)

    r2 = F2_ppf(u2, alpha, beta, gamma)
    return r2

# ------------------------- Exercise 10 functions -------------------------
def clayton_copula_uv(*, theta: float, size: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula (U1,U2) con cópula Clayton(theta) usando muestreo condicional.

    Fórmula (de tu Ej.10): si W ~ U(0,1) independiente de U1,
      U2 = ( 1 + U1^{-theta} * ( W^{-theta/(1+theta)} - 1 ) )^{-1/theta}.
    """
    if theta <= 0.0:
        raise ValueError("theta must be > 0.")
    rng = np.random.default_rng(seed)

    u1 = rng.random(size)
    w = rng.random(size)

    a = w ** (-theta / (1.0 + theta)) - 1.0
    u2 = (1.0 + (u1 ** (-theta)) * a) ** (-1.0 / theta)

    # robustez numérica
    eps = 1e-12
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    return u1, u2


def clayton_returns_from_pdfI(
    *,
    theta: float,
    size: int,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula (R1,R2) con dependencia Clayton(theta) y marginales del pdf (I):
      R1 = F1^{-1}(U1), R2 = F2^{-1}(U2).
    Devuelve (R1,R2,U1,U2).
    """
    u1, u2 = clayton_copula_uv(theta=theta, size=size, seed=seed)
    r1 = F1_ppf(u1, alpha, beta, gamma)
    r2 = F2_ppf(u2, alpha, beta, gamma)
    return r1, r2, u1, u2

def clayton_quantile_curve_r2(
    r1: np.ndarray,
    *,
    q: float,
    theta: float,
    alpha: float,
    beta: float,
    gamma: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Curva cuantil q para cópula Clayton condicional:

      u1 = F1(r1)
      u2 = ( 1 + u1^{-theta} * ( q^{-theta/(1+theta)} - 1 ) )^{-1/theta}
      r2 = F2^{-1}(u2)

    Marginales F1,F2 son las del pdf (I) implementadas previamente.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1).")
    if theta <= 0.0:
        raise ValueError("theta must be > 0.")

    u1 = F1_cdf(r1, alpha, beta, gamma)
    u1 = np.clip(u1, eps, 1.0 - eps)

    a = q ** (-theta / (1.0 + theta)) - 1.0
    u2 = (1.0 + (u1 ** (-theta)) * a) ** (-1.0 / theta)
    u2 = np.clip(u2, eps, 1.0 - eps)

    r2 = F2_ppf(u2, alpha, beta, gamma)
    return r2


def clayton_quantile_curve_u2(
    u1: np.ndarray, *, q: float, theta: float, eps: float = 1e-12
) -> np.ndarray:
    """
    Curva cuantil en el plano uniforme (u1,u2):
      u2(u1;q) = ( 1 + u1^{-theta} * ( q^{-theta/(1+theta)} - 1 ) )^{-1/theta}
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1).")
    if theta <= 0.0:
        raise ValueError("theta must be > 0.")
    u1 = np.asarray(u1, float)
    u1 = np.clip(u1, eps, 1.0 - eps)
    a = q ** (-theta / (1.0 + theta)) - 1.0
    u2 = (1.0 + (u1 ** (-theta)) * a) ** (-1.0 / theta)
    return np.clip(u2, eps, 1.0 - eps)

# ---------- Exercise 11 functions ----------

def inverse_triangular(u):
    """
    Transforma uniformes u en variables Triangulares(1, 5, 3).
    """
    x = np.zeros_like(u)
    # Máscara para el primer tramo [1, 3] donde area acumulada es 0.5
    mask1 = (u <= 0.5)
    # Máscara para el segundo tramo (3, 5]
    mask2 = (u > 0.5)
    
    # Aplicación de fórmulas inversas
    x[mask1] = 1 + np.sqrt(8 * u[mask1])
    x[mask2] = 5 - np.sqrt(8 * (1 - u[mask2]))
    return x

# ------------------------- Exercise 12 functions -------------------------
def t_copula_uv(*, rho: float, nu: int, size: int, seed: int | None = None):
    """
    Simula (U1,U2) ~ t-copula con correlación rho y dof nu:
      z ~ N(0, I), x = L z con cov(x)=Psi
      s ~ chi2_nu indep
      y = sqrt(nu/s) * x  -> multivar t
      u_i = T_nu(y_i)
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("Require -1 < rho < 1.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    rng = np.random.default_rng(seed)

    Psi = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Psi)

    z = rng.standard_normal((size, 2))
    x = z @ L.T

    s = chi2.rvs(df=nu, size=size, random_state=rng)
    y = x * np.sqrt(nu / s)[:, None]

    u = t.cdf(y, df=nu)
    return u[:, 0], u[:, 1]


# ------------------------------- Exercise 16 functions -------------------------------
def C_G_cond(u2: float, u1: float, rho: float) -> float:
    z2 = norm.ppf(u2)
    z1 = norm.ppf(u1)
    return norm.cdf((z2 - rho * z1) / np.sqrt(1.0 - rho**2))


def C_NM_cond(u2: float, u1: float, *, pi: float, rho1: float, rho2: float) -> float:
    return (
        pi * C_G_cond(u2, u1, rho1)
        + (1.0 - pi) * C_G_cond(u2, u1, rho2)
    )

def solve_u2_for_quantile(
    u1: float,
    q: float,
    *,
    pi: float,
    rho1: float,
    rho2: float,
    tol: float = 1e-10,
    maxiter: int = 100
) -> float:

    def f(u2: float) -> float:
        return C_NM_cond(u2, u1, pi=pi, rho1=rho1, rho2=rho2) - q

    lo, hi = 1e-12, 1.0 - 1e-12
    flo, fhi = f(lo), f(hi)

    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        if abs(fmid) < tol:
            return mid

        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)

def simulate_nm_copula(N, *, pi, rho1, rho2, seed=None):
    rng = np.random.default_rng(seed)

    # seleccionar componente
    comp = rng.uniform(size=N) < pi

    U = np.empty((N, 2))

    for k, rho in enumerate([rho1, rho2]):
        idx = (comp if k == 0 else ~comp)
        n_k = np.sum(idx)
        if n_k == 0:
            continue

        Psi = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(Psi)

        z = rng.standard_normal((n_k, 2))
        x = z @ L.T
        U[idx] = norm.cdf(x)

    return U[:, 0], U[:, 1]

# ------------------------------- Exercise 17 functions -------------------------------

def simulate_truncated(u, p):
    """Genera variable normal truncada dado uniforme u y params p"""
    # Estandarizar límites
    a_std = (p['a'] - p['mu']) / p['sigma']
    b_std = (p['b'] - p['mu']) / p['sigma']
    
    phi_a = norm.cdf(a_std)
    phi_b = norm.cdf(b_std)
    
    # Interpolación
    term = phi_a + u * (phi_b - phi_a)
    
    # Inversa Phi^-1
    # Clip para estabilidad numérica (evitar log(0) o inf)
    term = np.clip(term, 1e-9, 1 - 1e-9)
    return p['mu'] + p['sigma'] * norm.ppf(term)