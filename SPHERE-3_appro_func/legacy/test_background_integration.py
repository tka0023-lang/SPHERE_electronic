#!/usr/bin/env python3
"""
Test script to verify background event integration functionality.
This script tests the core functions without running the full pipeline.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    _load_background_events,
    _merge_signal_and_background,
    load_data,
    DEFAULT_MOSHITS_BG_BASE_DIR,
    DEFAULT_MOSHITS_BASE_DIR
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_load_background_events():
    """Test loading background events into memory."""
    logger.info("=" * 60)
    logger.info("TEST 1: Loading background events")
    logger.info("=" * 60)

    bg_data = _load_background_events(DEFAULT_MOSHITS_BG_BASE_DIR)

    if bg_data:
        logger.info(f"✓ Successfully loaded {len(bg_data)} background events")
        # Check first event structure
        if len(bg_data) > 0:
            data_pix, data_seg = bg_data[0]
            logger.info(f"  First event has {len(data_pix)} pixel hits")
            logger.info(f"  First event has {len(data_seg)} segment hits")
        return True
    else:
        logger.error("✗ Failed to load background events")
        return False


def test_merge_signal_and_background():
    """Test merging signal and background events."""
    logger.info("=" * 60)
    logger.info("TEST 2: Merging signal and background events")
    logger.info("=" * 60)

    # Load a sample signal event
    signal_dir = DEFAULT_MOSHITS_BASE_DIR / 'moshits_p'
    if not signal_dir.exists():
        logger.warning(f"Signal directory not found: {signal_dir}")
        logger.info("Skipping merge test")
        return True

    # Get first signal file
    signal_files = [f for f in os.listdir(signal_dir) if Path(f).suffix == '']
    if not signal_files:
        logger.warning(f"No signal files found in {signal_dir}")
        logger.info("Skipping merge test")
        return True

    signal_file = signal_dir / signal_files[0]
    logger.info(f"Loading signal event from: {signal_file}")

    try:
        signal_data_pix, signal_data_seg = load_data(signal_file)
        logger.info(f"  Signal event: {len(signal_data_pix)} pixel hits")

        # Load background events
        bg_data = _load_background_events(DEFAULT_MOSHITS_BG_BASE_DIR)
        if not bg_data:
            logger.warning("No background events available")
            return True

        bg_data_pix, bg_data_seg = bg_data[0]
        logger.info(f"  Background event: {len(bg_data_pix)} pixel hits")

        # Merge
        merged_pix, merged_seg = _merge_signal_and_background(
            signal_data_pix, signal_data_seg, bg_data_pix, bg_data_seg
        )

        expected_pix = len(signal_data_pix) + len(bg_data_pix)
        expected_seg = len(signal_data_seg) + len(bg_data_seg)

        logger.info(f"  Merged event: {len(merged_pix)} pixel hits (expected: {expected_pix})")
        logger.info(f"  Merged event: {len(merged_seg)} segment hits (expected: {expected_seg})")

        if len(merged_pix) == expected_pix and len(merged_seg) == expected_seg:
            logger.info("✓ Merge successful - pixel counts match")
            return True
        else:
            logger.error("✗ Merge failed - pixel counts don't match")
            return False

    except Exception as e:
        logger.error(f"✗ Error during merge test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist."""
    logger.info("=" * 60)
    logger.info("TEST 0: Checking directory structure")
    logger.info("=" * 60)

    bg_dir = DEFAULT_MOSHITS_BG_BASE_DIR
    logger.info(f"Background directory: {bg_dir}")

    if not bg_dir.exists():
        logger.error(f"✗ Background directory does not exist: {bg_dir}")
        return False

    logger.info(f"✓ Background directory exists")

    # Count files
    bg_files = [f for f in os.listdir(bg_dir) if Path(f).suffix == '']
    logger.info(f"  Found {len(bg_files)} background files")

    return True


if __name__ == '__main__':
    logger.info("\n" + "=" * 60)
    logger.info("BACKGROUND EVENT INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")

    results = []

    # Test 0: Directory structure
    results.append(("Directory structure", test_directory_structure()))

    # Test 1: Load background events
    results.append(("Load background events", test_load_background_events()))

    # Test 2: Merge events
    results.append(("Merge signal and background", test_merge_signal_and_background()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)
