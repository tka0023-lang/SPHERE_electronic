# sphere_appro/background.py
import logging
import random
from pathlib import Path
import numpy as np

from .io_data import EventData, load_event_file, list_event_files

logger = logging.getLogger(__name__)


def compute_background_level(bg_dir: Path, n_pixels: int,
                             sample_size: int = 100) -> float:
    if not bg_dir.exists():
        logger.warning('Background directory not found: %s', bg_dir)
        return 0.0
    files = list_event_files(bg_dir)
    if not files:
        logger.warning('No background files in %s', bg_dir)
        return 0.0
    sample = random.sample(files, k=min(sample_size, len(files)))
    levels = []
    for f in sample:
        try:
            ev = load_event_file(f)
            levels.append(len(ev.abs_pix) / n_pixels)
        except Exception as e:
            logger.debug('Skip bg file %s: %s', f, e)
    if not levels:
        return 0.0
    level = float(np.median(levels))
    logger.info('bg_level=%.6f from %d/%d files', level, len(levels), len(sample))
    return level


def merge_signal_and_background(signal: EventData, bg: EventData) -> EventData:
    return EventData(
        seg=np.concatenate([signal.seg, bg.seg]),
        pix=np.concatenate([signal.pix, bg.pix]),
        abs_pix=np.concatenate([signal.abs_pix, bg.abs_pix]),
    )
