# sphere_appro/event.py
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import Config
from .io_data import (
    create_shared_geometry, list_event_files, list_event_ids,
    save_results_csv, load_detector_geometry, build_ldf, EventData,
    discover_bin_files, BinFileInfo,
    discover_moshit_zst_dirs, list_moshit_zst_files,
)
from .config import RESULT_COLUMNS as BASE_RESULT_COLUMNS
from .background import compute_background_level
from .worker import (
    init_worker, process_file, process_event_bin, process_event_bin_flat,
    process_moshit_zst_flat,
)

logger = logging.getLogger(__name__)

PARTICLE_FOLDERS = ['moshits_p', 'moshits_N', 'moshits_Fe']


class Pipeline:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        cfg = self.config

        # Smoke test mode
        if cfg.smoke:
            self._smoke_test()
            return

        # Hierarchical data mode
        if cfg.data_root is not None:
            self._run_data_dir()
            return

        # Set fork method for safety with shared memory
        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            pass  # already set

        # Create shared geometry
        shared_geom = create_shared_geometry(cfg.pixel_path)
        logger.info('Shared geometry created')

        # Compute background level
        geom = load_detector_geometry(cfg.pixel_path)
        n_pixels = len(geom.pix_x)
        bg_level = compute_background_level(
            cfg.bg_root, n_pixels, sample_size=cfg.bg_sample,
        )
        logger.info('Background level: %.6f', bg_level)

        try:
            for folder in PARTICLE_FOLDERS:
                self._process_particle_type(
                    folder, shared_geom.meta, bg_level,
                )
        finally:
            shared_geom.release()
            logger.info('Shared memory released')

    def _process_particle_type(self, folder: str, geom_meta: dict,
                                bg_level: float):
        cfg = self.config
        moshits_dir = cfg.moshits_root / folder
        if not moshits_dir.exists():
            logger.warning('Directory not found: %s', moshits_dir)
            return

        # Auto-detect binary format
        bin_path = moshits_dir / "events.bin"
        use_binary = bin_path.exists()

        if use_binary:
            event_ids = list_event_ids(bin_path)
            if cfg.files_limit is not None:
                event_ids = event_ids[:cfg.files_limit]
            if not event_ids:
                logger.warning('No events in %s', bin_path)
                return
            total = len(event_ids)
            logger.info('Processing %s (binary): %d events with %d workers',
                         folder, total, cfg.workers)
        else:
            files = list_event_files(moshits_dir)
            if cfg.files_limit is not None:
                files = files[:cfg.files_limit]
            if not files:
                logger.warning('No files in %s', moshits_dir)
                return
            total = len(files)
            logger.info('Processing %s: %d files with %d workers',
                         folder, total, cfg.workers)

        results = []
        completed = 0

        with ProcessPoolExecutor(
            max_workers=cfg.workers,
            initializer=init_worker,
            initargs=(geom_meta, bg_level, cfg.min_intensity),
        ) as executor:
            if use_binary:
                futures = {executor.submit(process_event_bin, str(bin_path), eid): eid
                           for eid in event_ids}
            else:
                futures = {executor.submit(process_file, f): f for f in files}

            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    logger.info('%s: %d/%d processed', folder, completed, total)
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error('Future failed: %s', e)

        if not results:
            logger.warning('No results for %s', folder)
            return

        results_arr = np.array(results, dtype=np.float64)
        particle_type = folder.split('_')[1]  # 'p', 'N', or 'Fe'
        csv_path = f'{folder}_params.csv'
        save_results_csv(results_arr, BASE_RESULT_COLUMNS, csv_path)
        logger.info('Saved %d results to %s', len(results_arr), csv_path)

        if not cfg.skip_vis:
            try:
                from .visualization import create_visualizations
                create_visualizations(results_arr, folder, particle_type)
            except Exception as e:
                logger.warning('Visualization failed: %s', e)

    def _run_data_dir(self):
        """Process hierarchical data: events.bin or .moshit.zst files."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from multiprocessing import Pool

        cfg = self.config

        # Auto-detect format: try events.bin first, then .moshit.zst
        bin_files = discover_bin_files(cfg.data_root)
        moshit_dirs = discover_moshit_zst_dirs(cfg.data_root, exclude_energies=cfg.exclude_energies) if not bin_files else []

        if moshit_dirs and not bin_files:
            self._run_moshit_zst(moshit_dirs)
            return

        if not bin_files:
            logger.error('No binary files or .moshit.zst dirs found in %s', cfg.data_root)
            return
        if not bin_files:
            logger.error('No binary files found in %s', cfg.data_root)
            return

        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            pass

        shared_geom = create_shared_geometry(cfg.pixel_path)
        logger.info('Shared geometry created')

        total_files = len(bin_files)
        n_base = len(BASE_RESULT_COLUMNS)
        flush_interval = 50_000

        output_path = Path(cfg.output)
        parts_dir = output_path.parent / (output_path.stem + '_parts')
        parts_dir.mkdir(parents=True, exist_ok=True)
        total_written = 0
        batch_idx = 0
        batch_rows = []

        def _flush_batch():
            nonlocal total_written, batch_idx, batch_rows
            if not batch_rows:
                return
            float_data = np.array([row[:n_base] for row in batch_rows], dtype=np.float64)
            columns = {}
            for i, col_name in enumerate(BASE_RESULT_COLUMNS):
                columns[col_name] = float_data[:, i]
            columns['particle'] = [row[n_base] for row in batch_rows]
            columns['energy'] = [row[n_base + 1] for row in batch_rows]
            columns['angle'] = np.array([row[n_base + 2] for row in batch_rows], dtype=np.int32)
            columns['height'] = np.array([row[n_base + 3] for row in batch_rows], dtype=np.int32)
            batch_table = pa.table(columns)
            part_path = parts_dir / f'part_{batch_idx:04d}.parquet'
            pq.write_table(batch_table, part_path)
            total_written += len(batch_rows)
            batch_idx += 1
            logger.info('Saved batch %d: %d results to %s (total: %d)',
                        batch_idx, len(batch_rows), part_path, total_written)
            batch_rows.clear()

        def _task_generator():
            """Lazy generator — yields one task at a time, no list in memory."""
            for bf in bin_files:
                event_ids = list_event_ids(bf.path)
                if cfg.files_limit is not None:
                    event_ids = event_ids[:cfg.files_limit]
                for eid in event_ids:
                    yield (str(bf.path), eid, bf.particle, bf.energy, bf.angle, bf.height)

        # Count total events for progress logging
        total_events = 0
        for bf in bin_files:
            eids = list_event_ids(bf.path)
            if cfg.files_limit is not None:
                total_events += min(len(eids), cfg.files_limit)
            else:
                total_events += len(eids)
        logger.info('Total events to process: %d from %d files', total_events, total_files)

        try:
            with Pool(
                processes=cfg.workers,
                initializer=init_worker,
                initargs=(shared_geom.meta, 0.0, cfg.min_intensity),
                maxtasksperchild=1000,
            ) as pool:
                completed = 0
                for result in pool.imap_unordered(
                    process_event_bin_flat, _task_generator(), chunksize=50,
                ):
                    completed += 1
                    if completed % 500 == 0:
                        logger.info('Progress: %d/%d events', completed, total_events)
                    if result is not None:
                        batch_rows.append(result)
                        if len(batch_rows) >= flush_interval:
                            _flush_batch()

                _flush_batch()
        finally:
            shared_geom.release()
            logger.info('Shared memory released')

        if total_written == 0:
            logger.warning('No results produced')
            return

        # Merge all part files into one parquet
        part_files = sorted(parts_dir.glob('part_*.parquet'))
        tables = [pq.read_table(p) for p in part_files]
        merged = pa.concat_tables(tables)
        pq.write_table(merged, cfg.output)
        logger.info('Merged %d parts (%d results) into %s', len(part_files), len(merged), cfg.output)

        # Cleanup parts
        for p in part_files:
            p.unlink()
        parts_dir.rmdir()
        logger.info('Cleaned up part files')

    def _run_moshit_zst(self, moshit_dirs):
        """Process hierarchical .moshit.zst data."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from multiprocessing import Pool

        cfg = self.config

        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            pass

        shared_geom = create_shared_geometry(cfg.pixel_path)
        logger.info('Shared geometry created')

        n_base = len(BASE_RESULT_COLUMNS)
        flush_interval = 50_000

        output_path = Path(cfg.output)
        parts_dir = output_path.parent / (output_path.stem + '_parts')
        parts_dir.mkdir(parents=True, exist_ok=True)
        total_written = 0
        batch_idx = 0
        batch_rows = []

        def _flush_batch():
            nonlocal total_written, batch_idx, batch_rows
            if not batch_rows:
                return
            float_data = np.array([row[:n_base] for row in batch_rows], dtype=np.float64)
            columns = {}
            for i, col_name in enumerate(BASE_RESULT_COLUMNS):
                columns[col_name] = float_data[:, i]
            columns['particle'] = [row[n_base] for row in batch_rows]
            columns['energy'] = [row[n_base + 1] for row in batch_rows]
            columns['angle'] = np.array([row[n_base + 2] for row in batch_rows], dtype=np.int32)
            columns['height'] = np.array([row[n_base + 3] for row in batch_rows], dtype=np.int32)
            batch_table = pa.table(columns)
            part_path = parts_dir / f'part_{batch_idx:04d}.parquet'
            pq.write_table(batch_table, part_path)
            total_written += len(batch_rows)
            batch_idx += 1
            logger.info('Saved batch %d: %d results (total: %d)', batch_idx, len(batch_rows), total_written)
            batch_rows.clear()

        def _task_generator():
            for d in moshit_dirs:
                files = list_moshit_zst_files(d.path)
                if cfg.files_limit is not None:
                    files = files[:cfg.files_limit]
                for f in files:
                    yield (str(f), d.particle, d.energy, d.angle, d.height)

        # Count total files
        total_events = 0
        for d in moshit_dirs:
            n = len(list_moshit_zst_files(d.path))
            if cfg.files_limit is not None:
                n = min(n, cfg.files_limit)
            total_events += n
        logger.info('Total events to process: %d from %d directories', total_events, len(moshit_dirs))

        try:
            with Pool(
                processes=cfg.workers,
                initializer=init_worker,
                initargs=(shared_geom.meta, 0.0, cfg.min_intensity),
                maxtasksperchild=1000,
            ) as pool:
                completed = 0
                for result in pool.imap_unordered(
                    process_moshit_zst_flat, _task_generator(), chunksize=50,
                ):
                    completed += 1
                    if completed % 500 == 0:
                        logger.info('Progress: %d/%d events', completed, total_events)
                    if result is not None:
                        batch_rows.append(result)
                        if len(batch_rows) >= flush_interval:
                            _flush_batch()

                _flush_batch()
        finally:
            shared_geom.release()
            logger.info('Shared memory released')

        if total_written == 0:
            logger.warning('No results produced')
            return

        # Merge parts
        part_files = sorted(parts_dir.glob('part_*.parquet'))
        tables = [pq.read_table(p) for p in part_files]
        merged = pa.concat_tables(tables)
        pq.write_table(merged, cfg.output)
        logger.info('Merged %d parts (%d results) into %s', len(part_files), len(merged), cfg.output)

        for p in part_files:
            p.unlink()
        parts_dir.rmdir()
        logger.info('Cleaned up part files')

    def _smoke_test(self):
        """Quick check: load geometry, build synthetic LDF."""
        logger.info('Running smoke test...')
        geom = load_detector_geometry(self.config.pixel_path)
        logger.info('Loaded geometry: %d pixels, %d segments',
                     len(geom.pix_x), len(geom.seg_x))

        # Synthetic event
        rng = np.random.default_rng(42)
        n_hits = 100
        abs_pix = rng.integers(0, len(geom.pix_x), size=n_hits)
        ev = EventData(
            seg=abs_pix // 7,
            pix=abs_pix % 7,
            abs_pix=abs_pix,
        )
        ldf = build_ldf(ev, geom)
        logger.info('Built LDF: %d pixels, total I=%.1f, max I=%.1f',
                     len(ldf.I), ldf.I.sum(), ldf.I.max())
        logger.info('Smoke test PASSED')
