# cython: boundscheck=False, wraparound=False, cdivision=True
cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt, exp

cnp.import_array()


    
cpdef cnp.ndarray[double, ndim=1] ldf_model(
    double p0, double p1, double p2, double p3,
    double x0, double y0,
    double[:] x, double[:] y,
):
    cdef int n = x.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double r, z, omega1, omega2, core, tail
    cdef int i
    with nogil:
        for i in range(n):
            r = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0))
            z = (r - R_ch) / sw
            if z > 500.0:
                z = 500.0
            elif z < -500.0:
                z = -500.0
            omega1 = 1.0 / (1.0 + exp(z))
            omega2 = 1.0 - omega1
            core = p0**2 / (1.0 + p1 * r + p2 * r * r + p3 * r ** 1.5)**2
            out[i] = core
    return out    
    # для аппроксимации плоскостью
cpdef cnp.ndarray[double, ndim=1] plane_model(
    double A, double B, double C, double D, double[:] x, double[:] y,):
    cdef int n = x.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double  core
    cdef int i
    with nogil:
        for i in range(n):
            core = (-1)*(D+B*y+A*x)/C
            out[i] = core
    return out


cpdef double chi2_ndf(double[:] I, double[:] F, double min_error, int n_params=11):
    cdef int n = I.shape[0]
    cdef int ndf = n - n_params
    if ndf < 1:
        ndf = 1
    cdef double total = 0.0, sigma, diff
    cdef int i
    if n == 0:
        return 1e10
    with nogil:
        for i in range(n):
            sigma = sqrt(I[i]) if I[i] > min_error else sqrt(min_error)
            diff = F[i] - I[i]
            total += (diff / sigma) * (diff / sigma)
    return total / ndf


cpdef double ldf_integrand(
    double r, double p0, double p1, double p2, double p3
):
    cdef double  core

    core = p0**2 / (1.0 + p1 * r + p2 * r * r + p3 * r **1.5)**2
    return (core) * r



cpdef cnp.ndarray[double, ndim=1] ldf_residuals(
    double[:] params, double[:] x, double[:] y,
    double[:] I, double[:] sigma,
):
    cdef double p0 = params[0], p1 = params[1], p2 = params[2], p3 = params[3]
    cdef double x0 = params[9], y0 = params[10]
    cdef int n = x.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double r, z, omega1, omega2, core, tail, model
    cdef int i
    with nogil:
        for i in range(n):
            r = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0))
            core = p0**2 / (1.0 + p1 * r + p2 * r * r + p3 * r **1.5)**2
            model = core
            out[i] = (model - I[i]) / sigma[i]
    return out
