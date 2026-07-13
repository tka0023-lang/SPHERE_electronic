"""SPHERE-3 Approximation package."""

try:
    from .ldf_core import ldf_model, chi2_ndf, ldf_integrand, ldf_residuals
except ImportError:
    from ._ldf_core_fallback import ldf_model, chi2_ndf, ldf_integrand, ldf_residuals
