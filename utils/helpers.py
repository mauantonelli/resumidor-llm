import logging
import time
import sys


class Timer:
    def __init__(self):
        self._start = None
        self._elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def elapsed_str(self) -> str:
        if self._elapsed < 60:
            return f"{self._elapsed:.1f}s"
        minutes = int(self._elapsed // 60)
        seconds = self._elapsed % 60
        return f"{minutes}m {seconds:.1f}s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("resumidor")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def format_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
