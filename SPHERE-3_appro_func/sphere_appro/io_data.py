# sphere_appro/io_data.py
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
import logging
import numpy as np

from .config import (
    DETECTOR_FOCAL_LENGTH, PIXEL_SKIP, N_TOP_PEAKS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# PARTICLE ID MAPPING
# ============================================================================

PARTICLE_ID_MAP = {14: 'p', 1407: 'N', 5626: 'Fe'}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BinFileInfo:
    path: Path
    particle_id: int      # 14, 1407, 5626
    particle: str         # 'p', 'N', 'Fe'
    energy: str           # '5PeV', '10PeV', '30PeV'
    angle: int            # 5, 10, 15, 20, 25, 30
    height: int           # 500, 1000


@dataclass
class EventHeader:
    clone_num: int
    h: float
    x_center: float
    y_center: float
    event_num: int

@dataclass
class EventData:
    seg: np.ndarray       # int64
    pix: np.ndarray       # int64
    abs_pix: np.ndarray   # int64

@dataclass
class DetectorGeometry:
    pix_x: np.ndarray     # float64
    pix_y: np.ndarray     # float64
    seg_x: np.ndarray     # float64
    seg_y: np.ndarray     # float64

@dataclass
class LDF:
    I: np.ndarray         # float64
    x: np.ndarray         # float64
    y: np.ndarray         # float64

@dataclass
class SharedGeometry:
    meta: dict
    shm_blocks: dict

    def release(self):
        for shm in self.shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                continue

# ============================================================================
# FILE I/O
# ============================================================================

def load_event_header(file_path) -> EventHeader:
    with open(file_path, 'r') as f:
        values = f.readline().split()
    if len(values) < 5:
        raise ValueError(f'Expected >=5 values in header, got {len(values)}')
    return EventHeader(
        clone_num=int(values[0]),
        h=float(values[1]),
        x_center=float(values[2]),
        y_center=float(values[3]),
        event_num=int(values[4]),
    )

def load_event_file(file_path) -> EventData:
    raw = np.loadtxt(file_path, skiprows=1, usecols=(0, 1), dtype=np.int64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    seg = raw[:, 0]
    pix = raw[:, 1]
    abs_pix = seg * PIXEL_SKIP + pix
    return EventData(seg=seg, pix=pix, abs_pix=abs_pix)

# ============================================================================
# BINARY FORMAT I/O
# ============================================================================

from .moshits_binary import read_index, read_event_meta, read_event_hits, META_SIZE
import struct

_bin_index_cache: dict[str, dict[int, dict]] = {}


def _get_bin_index(path) -> dict[int, dict]:
    key = str(Path(path).resolve())
    if key not in _bin_index_cache:
        index = read_index(path)
        _bin_index_cache[key] = {e["event_id"]: e for e in index}
    return _bin_index_cache[key]


def load_event_header_bin(bin_path, event_id: int) -> EventHeader:
    lookup = _get_bin_index(bin_path)
    if event_id not in lookup:
        raise KeyError(f"Event {event_id} not found in {bin_path}")
    entry = lookup[event_id]
    meta = read_event_meta(bin_path, entry["offset"])
    return EventHeader(
        clone_num=meta["clone_num"],
        h=meta["h"],
        x_center=meta["x_center"],
        y_center=meta["y_center"],
        event_num=meta["event_num"],
    )


def load_event_file_bin(bin_path, event_id: int) -> EventData:
    lookup = _get_bin_index(bin_path)
    if event_id not in lookup:
        raise KeyError(f"Event {event_id} not found in {bin_path}")
    entry = lookup[event_id]
    hits = read_event_hits(bin_path, entry["offset"] + META_SIZE, entry["n_rows"])
    seg = hits["seg"].astype(np.int64)
    pix = hits["pix"].astype(np.int64)
    return EventData(seg=seg, pix=pix, abs_pix=seg * PIXEL_SKIP + pix)


def list_event_ids(bin_path) -> list[int]:
    lookup = _get_bin_index(bin_path)
    return list(lookup.keys())


# ============================================================================
# MOSHIT.ZST FORMAT I/O (SPHERE-3_G4 MoshitWriter output)
# ============================================================================

MOSH_HIT_DTYPE = np.dtype([
    ('pixel', '<u2'), ('origin', 'u1'), ('kk', 'u1'),
    ('ii', '<u2'), ('jj', '<u2'),
    ('t', '<f4'), ('t0', '<f4'),
])  # 16 bytes per hit


def load_moshit_zst(file_path) -> tuple[EventHeader, EventData]:
    """Read a .moshit.zst file (MOSH binary format, zstd-compressed)."""
    import zstandard as zstd

    with open(file_path, 'rb') as f:
        compressed = f.read()

    raw = zstd.ZstdDecompressor().decompress(compressed)

    if len(raw) < 24:
        raise ValueError(f'File too small: {len(raw)} bytes')
    magic = raw[:4]
    if magic != b'MOSH':
        raise ValueError(f'Invalid magic: {magic!r}')
    version = raw[4]
    if version != 1:
        raise ValueError(f'Unsupported version: {version}')

    zz, xsh, ysh, n_hits = struct.unpack_from('<fffI', raw, 8)

    header = EventHeader(clone_num=0, h=zz, x_center=xsh, y_center=ysh, event_num=0)

    if n_hits > 0:
        hits = np.frombuffer(raw, dtype=MOSH_HIT_DTYPE, count=n_hits, offset=24)
        abs_pix = hits['pixel'].astype(np.int64)
    else:
        abs_pix = np.array([], dtype=np.int64)

    seg = abs_pix // PIXEL_SKIP
    pix = abs_pix % PIXEL_SKIP
    return header, EventData(seg=seg, pix=pix, abs_pix=abs_pix)


@dataclass
class MoshitZstFileInfo:
    path: Path
    particle_id: int
    particle: str
    energy: str
    angle: int
    height: int


def discover_moshit_zst_dirs(
    data_root: Path,
    exclude_energies: frozenset[str] | set[str] = frozenset(),
) -> list[MoshitZstFileInfo]:
    """Scan data_root/{particle_id}/{energy}/{angle}/{height}/ for .moshit.zst files.

    Returns one entry per leaf directory (not per file).
    """
    data_root = Path(data_root)
    results = []
    for pid_dir in sorted(data_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        try:
            particle_id = int(pid_dir.name)
        except ValueError:
            continue
        if particle_id not in PARTICLE_ID_MAP:
            continue
        for energy_dir in sorted(pid_dir.iterdir()):
            if not energy_dir.is_dir():
                continue
            if energy_dir.name in exclude_energies:
                continue
            for angle_dir in sorted(energy_dir.iterdir()):
                if not angle_dir.is_dir():
                    continue
                for height_dir in sorted(angle_dir.iterdir()):
                    if not height_dir.is_dir():
                        continue
                    # Check if dir has .moshit.zst files
                    sample = next(height_dir.glob('*.moshit.zst'), None)
                    if sample is None:
                        continue
                    results.append(MoshitZstFileInfo(
                        path=height_dir,
                        particle_id=particle_id,
                        particle=PARTICLE_ID_MAP[particle_id],
                        energy=energy_dir.name,
                        angle=int(angle_dir.name),
                        height=int(height_dir.name),
                    ))
    logger.info('Discovered %d moshit.zst directories in %s', len(results), data_root)
    return results


def list_moshit_zst_files(directory: Path) -> list[Path]:
    """List all .moshit.zst files in a directory."""
    return sorted(directory.glob('*.moshit.zst'))


def discover_bin_files(data_root: Path) -> list[BinFileInfo]:
    """Scan data_root/{particle_id}/{energy}/{angle}/{height}/events.bin."""
    data_root = Path(data_root)
    results = []
    for bin_path in sorted(data_root.rglob('events.bin')):
        parts = bin_path.relative_to(data_root).parts
        # expect: (particle_id, energy, angle, height, 'events.bin')
        if len(parts) != 5:
            logger.warning('Unexpected path depth: %s', bin_path)
            continue
        pid_str, energy, angle_str, height_str, _ = parts
        particle_id = int(pid_str)
        if particle_id not in PARTICLE_ID_MAP:
            logger.warning('Unknown particle id %d in %s', particle_id, bin_path)
            continue
        results.append(BinFileInfo(
            path=bin_path,
            particle_id=particle_id,
            particle=PARTICLE_ID_MAP[particle_id],
            energy=energy,
            angle=int(angle_str),
            height=int(height_str),
        ))
    logger.info('Discovered %d binary files in %s', len(results), data_root)
    return results


# ============================================================================
# DETECTOR GEOMETRY
# ============================================================================

def load_detector_geometry(pixel_data_path) -> DetectorGeometry:
    raw = np.loadtxt(pixel_data_path)
    x = raw[:, 0] / raw[:, 2] * DETECTOR_FOCAL_LENGTH
    y = raw[:, 1] / raw[:, 2] * DETECTOR_FOCAL_LENGTH
    seg_x = x[::PIXEL_SKIP]
    seg_y = y[::PIXEL_SKIP]
    return DetectorGeometry(pix_x=x, pix_y=y, seg_x=seg_x, seg_y=seg_y)

def list_event_files(directory: Path) -> list[str]:
    if not directory.exists():
        logger.warning('Directory does not exist: %s', directory)
        return []
    files = []
    for f in directory.iterdir():
        if f.name.startswith('.') or f.suffix != '':
            continue
        files.append(str(f))
    return sorted(files)

# ============================================================================
# LDF CONSTRUCTION
# ============================================================================
def focal_to_flat_mosaic(xf, yf, F=330.0):
    zc = 971.41 - 862.5  # центр сферы мозаики по z, mm
    R = 862.5            # радиус сферы пикселей, mm

    vx = xf / F
    vy = yf / F
    vz = 1.0

    a = vx*vx + vy*vy + vz*vz
    b = -2.0 * zc * vz
    c = zc*zc - R*R

    t = (-b + (b*b - 4*a*c)**0.5) / (2*a)

    X = t * vx
    Y = t * vy
    Z = t * vz
    return X, Y, Z


def build_ldf(event: EventData, geometry: DetectorGeometry,
              which: str = 'pix', bg_level: float = 0.0) -> LDF:
    if which == 'pix':
        hits = np.bincount(event.abs_pix, minlength=len(geometry.pix_x))
        x, y = geometry.pix_x, geometry.pix_y
    elif which == 'seg':
        hits = np.bincount(event.seg, minlength=len(geometry.seg_x))
        x, y = geometry.seg_x, geometry.seg_y
    else:
        raise ValueError(f'Unknown geometry type: {which}')
    x,y,z=focal_to_flat_mosaic(x,y,F=330.0)
    I = hits.astype(np.float64) - bg_level
    np.clip(I, 0.0, None, out=I)
    return LDF(I=I, x=x, y=y)

def compute_peak_center(ldf: LDF, n_top: int = N_TOP_PEAKS) -> tuple[float, float]:
    if ldf.I.sum() == 0:
        return 0.0, 0.0
    n = min(n_top, len(ldf.I))
    top_idx = np.argpartition(ldf.I, -n)[-n:]
    return float(ldf.x[top_idx].mean()), float(ldf.y[top_idx].mean())

def compute_weighted_center(ldf: LDF) -> tuple[float, float]:
    total_I = ldf.I.sum()
    if total_I == 0:
        return 0.0, 0.0
    x_center = float(np.sqrt((ldf.x**2 * ldf.I).sum() / total_I))
    y_center = float(np.sqrt((ldf.y**2 * ldf.I).sum() / total_I))
    max_idx = int(ldf.I.argmax())
    if ldf.x[max_idx] < 0:
        x_center = -x_center
    if ldf.y[max_idx] < 0:
        y_center = -y_center
    return x_center, y_center

# ============================================================================
# CSV I/O
# ============================================================================

def save_results_csv(results: np.ndarray, columns: list[str], file_path: str):
    header = ','.join(columns)
    np.savetxt(file_path, results, delimiter=',', header=header, comments='')

def load_results_csv(file_path: str) -> tuple[np.ndarray, dict[str, int]]:
    with open(file_path) as f:
        header_line = f.readline().strip()
    names = header_line.split(',')
    col = {name: idx for idx, name in enumerate(names)}
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, col

# ============================================================================
# SHARED MEMORY
# ============================================================================

def create_shared_geometry(pixel_data_path: Path) -> SharedGeometry:
    geom = load_detector_geometry(pixel_data_path)
    shm_blocks = {}
    meta = {}
    for name, arr in [('pix_x', geom.pix_x), ('pix_y', geom.pix_y),
                       ('seg_x', geom.seg_x), ('seg_y', geom.seg_y)]:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
        shm_blocks[name] = shm
        meta[name] = {'name': shm.name, 'shape': arr.shape, 'dtype': str(arr.dtype)}
    return SharedGeometry(meta=meta, shm_blocks=shm_blocks)

def attach_shared_geometry(meta: dict) -> DetectorGeometry:
    arrays = {}
    for key, info in meta.items():
        shm = shared_memory.SharedMemory(name=info['name'])
        view = np.ndarray(info['shape'], dtype=np.dtype(info['dtype']), buffer=shm.buf)
        arrays[key] = view.copy()  # copy to own memory so shm can be closed
        shm.close()
    return DetectorGeometry(
        pix_x=arrays['pix_x'], pix_y=arrays['pix_y'],
        seg_x=arrays['seg_x'], seg_y=arrays['seg_y'],
    )
