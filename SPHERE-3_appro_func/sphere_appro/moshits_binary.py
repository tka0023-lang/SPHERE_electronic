"""
Binary format for SPHERE-3 moshits event data.

Format (all little-endian, packed):
  Header:  4s magic 'SPHR' | H version | I n_events
  Index:   N * (H event_id | Q offset | I n_rows)
  Events:  per event: metadata (ifffi) + hits (structured array)
"""
import struct
from pathlib import Path
import numpy as np

MAGIC = b"SPHR"
FORMAT_VERSION = 1

HEADER_FMT = "<4sHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 10

INDEX_ENTRY_FMT = "<HQI"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FMT)  # 14

META_FMT = "<ifffi"
META_SIZE = struct.calcsize(META_FMT)  # 20

HIT_DTYPE = np.dtype([
    ("seg", "<u2"), ("pix", "<u2"),
    ("ii", "<u2"),  ("jj", "<u2"),
    ("t", "<f4"),   ("tt", "<f4"),
])  # 16 bytes per row


def write_events_bin(path, events):
    path = Path(path)
    n_events = len(events)

    index_end = HEADER_SIZE + n_events * INDEX_ENTRY_SIZE
    offsets = []
    current = index_end
    for _, _, hits in events:
        offsets.append(current)
        current += META_SIZE + hits.nbytes

    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, n_events))
        for i, (ev_id, _, hits) in enumerate(events):
            f.write(struct.pack(INDEX_ENTRY_FMT, ev_id, offsets[i], len(hits)))
        for (_, meta, hits), offset in zip(events, offsets):
            assert f.tell() == offset
            f.write(struct.pack(META_FMT,
                                meta["clone_num"], meta["h"],
                                meta["x_center"], meta["y_center"],
                                meta["event_num"]))
            if len(hits) > 0:
                f.write(hits.tobytes())


def read_index(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        magic, version, n_events = struct.unpack(HEADER_FMT, raw)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        index = []
        for _ in range(n_events):
            raw = f.read(INDEX_ENTRY_SIZE)
            ev_id, offset, n_rows = struct.unpack(INDEX_ENTRY_FMT, raw)
            index.append({"event_id": ev_id, "offset": offset, "n_rows": n_rows})
    return index


def read_event_meta(path, offset):
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(META_SIZE)
    clone_num, h, x_center, y_center, event_num = struct.unpack(META_FMT, raw)
    return {
        "clone_num": clone_num, "h": h,
        "x_center": x_center, "y_center": y_center,
        "event_num": event_num,
    }


def read_event_hits(path, offset, n_rows):
    if n_rows == 0:
        return np.zeros(0, dtype=HIT_DTYPE)
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(n_rows * HIT_DTYPE.itemsize)
    return np.frombuffer(raw, dtype=HIT_DTYPE)
