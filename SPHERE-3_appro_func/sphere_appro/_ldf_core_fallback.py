# sphere_appro/_ldf_core_fallback.py
"""Pure numpy fallback — adapted F_new model (Latypova et al. 2022)."""
# ruff: noqa: E741 — I is standard physics notation for intensity
import numpy as np


def ldf_model(p0, p1, p2, p3, p4, p5, p6, R_ch, sw, x0, y0, x, y):
    """Two-component rational LDF with sigmoid crossover.

    F(r) = p0/(1+p1·r+p2·r²+p3·r³) · ω₁  +  p4/(1+p5·r+p6·r²) · ω₂
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    core = p0**2 / (1.0 + p1 * r + p2 * r**2 + p3 * r**1.5)**2

    return core 
#
def plane_model(A, B, C, D, x, y):

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    core = (-1)*(D+B*y+A*x)/C

    return core 


def chi2_ndf(I, F, min_error, n_params=11):
    """Reduced chi-squared: χ²/(N-k)."""
    I = np.asarray(I, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    sigma = np.sqrt(np.maximum(I, min_error))
    ndf = max(I.size - n_params, 1)
    return float(np.sum(((F - I) / sigma) ** 2) / ndf)


def ldf_integrand(r, p0, p1, p2, p3, p4, p5, p6, R_ch, sw):
    """Integrand for radial integration: F(r) * r."""


    core = p0**2 / (1.0 + p1 * r + p2 * r**2 + p3 * r**1.5)**2


    return float((core) * r)









def ldf_residuals(params, x, y, I, sigma):
    """Weighted residuals for least-squares (not used with Minuit, kept for compatibility)."""
    p0, p1, p2, p3, p4, p5, p6, R_ch, sw, x0, y0 = params
    model = ldf_model(p0, p1, p2, p3, p4, p5, p6, R_ch, sw, x0, y0, x, y)
    return (np.asarray(model) - np.asarray(I)) / np.asarray(sigma)
