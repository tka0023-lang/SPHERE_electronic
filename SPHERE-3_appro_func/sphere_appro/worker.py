# sphere_appro/worker.py
import math
import logging
from typing import Optional

from .config import MAX_CENTER_DISTANCE, MAX_PEAK_DISTANCE
from .io_data import (
    DetectorGeometry, attach_shared_geometry,
    load_event_header, load_event_file, build_ldf, compute_peak_center,
    load_event_header_bin, load_event_file_bin,
    load_moshit_zst,
)
from .optimizer import fit_ldf, integrate_ldf

logger = logging.getLogger(__name__)

# Global worker state
_geometry: Optional[DetectorGeometry] = None
_bg_level: float = 0.0
_min_intensity: float = 10.0


def init_worker(geometry_shm_meta, bg_level, min_intensity):
    global _geometry, _bg_level, _min_intensity
    _geometry = attach_shared_geometry(geometry_shm_meta)
    _bg_level = bg_level
    _min_intensity = min_intensity
    logger.debug('Worker initialized: bg_level=%.6f', bg_level)


def process_file(file_path: str) -> Optional[tuple]:
    global _geometry, _bg_level, _min_intensity

    try:
        header = load_event_header(file_path)
        center_dist = math.hypot(header.x_center, header.y_center)
        if center_dist > MAX_CENTER_DISTANCE:
            return None

        event = load_event_file(file_path)
        ldf = build_ldf(event, _geometry, bg_level=_bg_level)

        total_I = float(ldf.I.sum())
        if total_I < _min_intensity:
            return None

        x_peak, y_peak = compute_peak_center(ldf)
        if math.hypot(x_peak, y_peak) > MAX_PEAK_DISTANCE:
            return None

        fit = fit_ldf(ldf.I, ldf.x, ldf.y, x_peak, y_peak)
        if fit is None:
            return None

        integral, int_err = integrate_ldf(
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw,
        )

        return (
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw, fit.x0, fit.y0,
            fit.fval, fit.chi2_ndf, fit.max_abs_d, fit.mean_abs_d,
            center_dist, float(ldf.I.max()), total_I,
            integral, int_err,fit.A,fit.B,fit.C,fit.D,fit.COSpl
        )
    except Exception as e:
        logger.error('Error processing %s: %s', file_path, e)
        return None


def process_moshit_zst(file_path: str) -> Optional[tuple]:
    """Process a single .moshit.zst file (SPHERE-3_G4 output)."""
    global _geometry, _bg_level, _min_intensity

    try:
        header, event = load_moshit_zst(file_path)
        center_dist = math.hypot(header.x_center, header.y_center)
        if center_dist > MAX_CENTER_DISTANCE:
            return None

        ldf = build_ldf(event, _geometry, bg_level=_bg_level)

        total_I = float(ldf.I.sum())
        if total_I < _min_intensity:
            return None

        x_peak, y_peak = compute_peak_center(ldf)
        if math.hypot(x_peak, y_peak) > MAX_PEAK_DISTANCE:
            return None

        fit = fit_ldf(ldf.I, ldf.x, ldf.y, x_peak, y_peak)
        if fit is None:
            return None

        integral, int_err = integrate_ldf(
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw,
        )

        return (
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw, fit.x0, fit.y0,
            fit.fval, fit.chi2_ndf, fit.max_abs_d, fit.mean_abs_d,
            center_dist, float(ldf.I.max()), total_I,
            integral, int_err, fit.A, fit.B, fit.C, fit.D, fit.COSpl
        )
    except Exception as e:
        logger.error('Error processing %s: %s', file_path, e)
        return None


def process_moshit_zst_flat(args: tuple) -> Optional[tuple]:
    """Wrapper for imap_unordered: (file_path, particle, energy, angle, height)."""
    file_path, particle, energy, angle, height = args
    result = process_moshit_zst(file_path)
    if result is None:
        return None
    return (*result, particle, energy, angle, height)


def process_event_bin_flat(args: tuple) -> Optional[tuple]:
    """Wrapper for imap_unordered (single-argument callable).

    Args is (bin_path, event_id, particle, energy, angle, height).
    Returns (*fit_result, particle, energy, angle, height) or None.
    """
    bin_path, event_id, particle, energy, angle, height = args
    result = process_event_bin(bin_path, event_id)
    if result is None:
        return None
    return (*result, particle, energy, angle, height)


def process_event_bin(bin_path: str, event_id: int) -> Optional[tuple]:
    """Process a single event from binary block. Same logic as process_file."""
    global _geometry, _bg_level, _min_intensity

    try:
        header = load_event_header_bin(bin_path, event_id)
        center_dist = math.hypot(header.x_center, header.y_center)
        if center_dist > MAX_CENTER_DISTANCE:
            return None

        event = load_event_file_bin(bin_path, event_id)
        ldf = build_ldf(event, _geometry, bg_level=_bg_level)

        total_I = float(ldf.I.sum())
        if total_I < _min_intensity:
            return None

        x_peak, y_peak = compute_peak_center(ldf)
        if math.hypot(x_peak, y_peak) > MAX_PEAK_DISTANCE:
            return None

        fit = fit_ldf(ldf.I, ldf.x, ldf.y, x_peak, y_peak)
        if fit is None:
            return None
        fit_pl=fit_pl

        integral, int_err = integrate_ldf(
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw,
        )

        return (
            fit.p0, fit.p1, fit.p2, fit.p3,
            fit.p4, fit.p5, fit.p6,
            fit.R_ch, fit.sw, fit.x0, fit.y0,
            fit.fval, fit.chi2_ndf, fit.max_abs_d, fit.mean_abs_d,
            center_dist, float(ldf.I.max()), total_I,
            integral, int_err, fit.A, fit.B, fit.C, fit.D, fit.COSpl
        )
    except Exception as e:
        logger.error('Error processing event %d from %s: %s', event_id, bin_path, e)
        return None
