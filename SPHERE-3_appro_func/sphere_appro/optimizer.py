# sphere_appro/optimizer.py
# ruff: noqa: E741 — I is standard physics notation for intensity
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad, IntegrationWarning

from .config import (
    LIMIT_P0_FACTOR,
    LIMIT_P1_MIN, LIMIT_P1_MAX,
    LIMIT_P2_MIN, LIMIT_P2_MAX,
    LIMIT_P3_MIN, LIMIT_P3_MAX,
    LIMIT_P4_MIN, LIMIT_P4_MAX,
    LIMIT_P5_MIN, LIMIT_P5_MAX,
    LIMIT_P6_MIN, LIMIT_P6_MAX,
    LIMIT_RCH_MIN, LIMIT_RCH_MAX,
    LIMIT_SW_MIN, LIMIT_SW_MAX,
    LIMIT_X0_MIN, LIMIT_X0_MAX, LIMIT_Y0_MIN, LIMIT_Y0_MAX,
    INTEGRATION_LOWER_BOUND, INTEGRATION_UPPER_BOUND,
    INTEGRATION_ABS_ERROR, INTEGRATION_REL_ERROR,
    MINUIT_STRATEGY_FAST, MINUIT_STRATEGY_PRECISE, MINUIT_CHI2_THRESHOLD,
)

logger = logging.getLogger(__name__)

try:
    from .ldf_core import ldf_model, chi2_ndf, ldf_integrand, plane_model
except ImportError:
    from ._ldf_core_fallback import ldf_model, chi2_ndf, ldf_integrand, plane_model


N_FREE_PARAMS = 11


@dataclass
class FitResult:
    p0: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    p6: float
    R_ch: float
    sw: float
    x0: float
    y0: float
    fval: float
    chi2_ndf: float
    max_abs_d: float
    mean_abs_d: float
    success: bool
    n_attempts: int
    A:float
    B:float
    C:float
    D:float
    COSpl:float


def estimate_params_adaptive(r, I):
    """Estimate initial parameters for F_new from radial profile moments."""
    valid = (I > 0) & np.isfinite(r) & np.isfinite(I)
    if not np.any(valid):
        return 1e6, 0.05, -1e-3, 1e-6, 1e4, -0.01, 1e-5, 100.0, 8.0

    r_v, I_v = r[valid], I[valid]
    I_max = float(I_v.max())
    I_norm = I_v / I_v.sum()

    # R_ch: radius where log-derivative changes most
    r_mean = float(np.sum(r_v * I_norm))
    R_ch = np.clip(r_mean, LIMIT_RCH_MIN + 5, LIMIT_RCH_MAX - 5)

    # Core region: estimate p1 from half-max radius
    half_max_mask = I_v >= 0.5 * I_max
    if np.sum(half_max_mask) > 1:
        r_half = float(r_v[half_max_mask].max())
        p1 = 1.0 / max(r_half, 5.0)
    else:
        core_mask = r_v < R_ch
        if np.sum(core_mask) > 3:
            r_core = r_v[core_mask]
            I_core_norm = I_v[core_mask] / I_v[core_mask].sum()
            r_c_mean = float(np.sum(r_core * I_core_norm))
            r_c_var = float(np.sum((r_core - r_c_mean) ** 2 * I_core_norm))
            width = max(np.sqrt(r_c_var), 5.0)
            p1 = 1.0 / width
        else:
            p1 = 0.05
    p2 = -p1 / (2.0 * R_ch)  # mild curvature
    p3 = abs(p2) / (3.0 * R_ch)  # small cubic correction

    # Tail region (r > R_ch)
    tail_mask = r_v >= R_ch
    if np.sum(tail_mask) > 2:
        r_tail = r_v[tail_mask]
        I_tail = I_v[tail_mask]
        I_tail_norm = I_tail / I_tail.sum()
        r_t_mean = float(np.sum(r_tail * I_tail_norm))
        width_t = max(r_t_mean - R_ch, 10.0)
        p5 = -1.0 / width_t
    else:
        p5 = -0.01
    p6 = abs(p5) / (2.0 * max(r_v.max(), 100.0))

    p0 = I_max
    p4 = float(I_v[tail_mask].max()) if np.any(tail_mask) else I_max * 0.01

    sw = 8.0

    return p0, p1, p2, p3, p4, p5, p6, R_ch, sw


def _scan_center(x, y, I, x_init, y_init):
    """Coarse scan to pick the best starting center."""
    nz = I > 0
    if not np.any(nz):
        return x_init, y_init
    cx = float(np.average(x[nz], weights=np.maximum(I[nz], 1e-6)))
    cy = float(np.average(y[nz], weights=np.maximum(I[nz], 1e-6)))
    centers = [(x_init, y_init), (cx, cy)]
    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    best_center, best_val = (x_init, y_init), float('inf')

    r_all = np.sqrt((x - x_init) ** 2 + (y - y_init) ** 2)
    est = estimate_params_adaptive(r_all, I)
    p0_t, p1_t, p2_t, p3_t, p4_t, p5_t, p6_t, R_ch_t, sw_t = est

    for base in centers:
        for dx, dy in offsets:
            xc, yc = base[0] + dx, base[1] + dy
            pred = np.asarray(ldf_model(
                p0_t, p1_t, p2_t, p3_t, p4_t, p5_t, p6_t,
                R_ch_t, sw_t, xc, yc, x, y,
            ))
            val = float(np.sqrt(np.mean((pred - I) ** 2)))
            if val < best_val:
                best_val = val
                best_center = (xc, yc)
    return best_center


def _compute_diagnostics(I_nz, F_nz):
    """Compute diagnostic metrics: max_abs_d, mean_abs_d."""
    safe_I = np.maximum(I_nz, 1e-12)
    d = (I_nz - F_nz) / safe_I
    return float(np.max(np.abs(d))), float(np.mean(np.abs(d)))


def _fit_minuit(I_nz, x_nz, y_nz, x_center, y_center,
                Imax, n_restarts=3) -> Optional[FitResult]:
    from iminuit import Minuit

    r_all = np.sqrt((x_nz - x_center) ** 2 + (y_nz - y_center) ** 2)
    best_result, best_chi2 = None, float('inf')

    for attempt in range(n_restarts):
        try:
            est = estimate_params_adaptive(r_all, I_nz)
            p0_i, p1_i, p2_i, p3_i, p4_i, p5_i, p6_i, R_ch_i, sw_i = est

            # Scale initial params across restarts
            if attempt == 0:
                pass  # use base estimates
            elif attempt == 1:
                p0_i *= 1.2
                p1_i *= 2.0
                R_ch_i *= 0.7
            else:
                p0_i *= 0.8
                p1_i *= 3.0
                p2_i *= 2.0
                R_ch_i *= 0.5

            def error_fn(p0, p1, p2, p3, p4, p5, p6, R_ch, sw, x0, y0):
                F = np.asarray(ldf_model(
                    p0, p1, p2, p3, p4, p5, p6, R_ch, sw, x0, y0,
                    x_nz, y_nz,
                ))
                return float(chi2_ndf(I_nz, F, 1.0, N_FREE_PARAMS))
            def error_pl(A, B, C, D):
                F = np.asarray(plane_model(
                    A, B, C, D, x_nz, y_nz,
                ))
                return float(chi2_ndf(I_nz, F, 1.0, N_FREE_PARAMS))

            m = Minuit(error_fn,
                       p0=p0_i, p1=p1_i, p2=p2_i, p3=p3_i,
                       p4=p4_i, p5=p5_i, p6=p6_i,
                       R_ch=R_ch_i, sw=sw_i,
                       x0=x_center, y0=y_center)           
            m1 = Minuit(error_pl,
                       A=1, B=1, C=1, D=50)
            m1.migrad()
            #находим угол от аппроксимации плоскостью
            COSpl=m1.values['C']/(m1.values['A']**2+m1.values['B']**2+m1.values['C']**2)**0.5

            # Adaptive strategy
            if attempt < 2 or best_chi2 <= MINUIT_CHI2_THRESHOLD:
                m.strategy = MINUIT_STRATEGY_FAST
            else:
                m.strategy = MINUIT_STRATEGY_PRECISE

            # Set limits
            p0_cap = LIMIT_P0_FACTOR * max(Imax, 1.0)
            m.limits['p0'] = (0, p0_cap)
            m.limits['p1'] = (LIMIT_P1_MIN, LIMIT_P1_MAX)
            m.limits['p2'] = (LIMIT_P2_MIN, LIMIT_P2_MAX)
            m.limits['p3'] = (LIMIT_P3_MIN, LIMIT_P3_MAX)
            m.limits['p4'] = (LIMIT_P4_MIN, LIMIT_P4_MAX)
            m.limits['p5'] = (LIMIT_P5_MIN, LIMIT_P5_MAX)
            m.limits['p6'] = (LIMIT_P6_MIN, LIMIT_P6_MAX)
            m.limits['R_ch'] = (LIMIT_RCH_MIN, LIMIT_RCH_MAX)
            m.limits['sw'] = (LIMIT_SW_MIN, LIMIT_SW_MAX)
            m.limits['x0'] = (LIMIT_X0_MIN, LIMIT_X0_MAX)
            m.limits['y0'] = (LIMIT_Y0_MIN, LIMIT_Y0_MAX)

            # Stage 1: fit primary params with secondary fixed
            m.fixed['p4'] = True
            m.fixed['p5'] = True
            m.fixed['p6'] = True
            m.fixed['sw'] = True
            m.fixed['x0'] = True
            m.fixed['y0'] = True
            m.simplex().migrad()

            # Stage 2: release all parameters
            m.fixed = False
            m.simplex().migrad()


            v = m.values
            v1 = m1.values
            F = np.asarray(ldf_model(
                v['p0'], v['p1'], v['p2'], v['p3'],
                v['p4'], v['p5'], v['p6'],
                v['R_ch'], v['sw'], v['x0'], v['y0'],
                x_nz, y_nz,
            ))
            chi2 = float(chi2_ndf(I_nz, F, 1.0, N_FREE_PARAMS))
            max_d, mean_d = _compute_diagnostics(I_nz, F)

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_result = FitResult(
                    p0=float(v['p0']), p1=float(v['p1']),
                    p2=float(v['p2']), p3=float(v['p3']),
                    p4=float(v['p4']), p5=float(v['p5']),
                    p6=float(v['p6']),
                    R_ch=float(v['R_ch']), sw=float(v['sw']),
                    x0=float(v['x0']), y0=float(v['y0']),
                    fval=float(m.fval),
                    chi2_ndf=chi2, max_abs_d=max_d, mean_abs_d=mean_d,
                    success=bool(m.valid), n_attempts=attempt + 1,
                    A=float(v1['A']),B=float(v1['B']),C=float(v1['C']),D=float(v1['D']),
                    COSpl=COSpl
                )
        except Exception as e:
            logger.warning('minuit attempt %d failed: %s', attempt + 1, e)

    return best_result


def fit_ldf(I, x, y, x_center, y_center,
            n_restarts=3) -> Optional[FitResult]:
    """Fit the F_new LDF model using Minuit."""
    I = np.asarray(I, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nz_mask = I > 0
    I_nz, x_nz, y_nz = I[nz_mask], x[nz_mask], y[nz_mask]
    if I_nz.size == 0:
        return None

    Imax = float(I_nz.max())
    x_center, y_center = _scan_center(x, y, I, x_center, y_center)

    return _fit_minuit(I_nz, x_nz, y_nz, x_center, y_center, Imax, n_restarts)


def integrate_ldf(p0, p1, p2, p3, p4, p5, p6, R_ch, sw,
                  a=INTEGRATION_LOWER_BOUND,
                  b=INTEGRATION_UPPER_BOUND) -> tuple[float, float]:
    """Integrate 2pi int F(r)*r dr over [a, b]."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            val, err = quad(
                lambda r: ldf_integrand(r, p0, p1, p2, p3, p4, p5, p6, R_ch, sw),
                a, b,
                epsabs=INTEGRATION_ABS_ERROR,
                epsrel=INTEGRATION_REL_ERROR,
            )
            return float(2 * np.pi * val), float(2 * np.pi * err)
        except (IntegrationWarning, Exception):
            return float('nan'), float('nan')
