import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import Timer, format_size, setup_logging


def test_format_size_bytes():
    assert format_size(500) == "500.0 B"


def test_format_size_kilobytes():
    assert format_size(1024) == "1.0 KB"


def test_format_size_megabytes():
    assert format_size(1024 * 1024) == "1.0 MB"


def test_format_size_gigabytes():
    assert format_size(1024 ** 3) == "1.0 GB"


def test_format_size_terabytes():
    assert format_size(1024 ** 4) == "1.0 TB"


def test_timer_records_time():
    with Timer() as t:
        time.sleep(0.1)
    assert t.elapsed >= 0.05
    assert t.elapsed < 2.0


def test_timer_elapsed_str_seconds():
    with Timer() as t:
        time.sleep(0.05)
    assert "s" in t.elapsed_str


def test_timer_initial_elapsed():
    t = Timer()
    assert t.elapsed == 0.0


def test_setup_logging():
    logger = setup_logging()
    assert logger.name == "resumidor"
    assert len(logger.handlers) > 0
