"""Utility helpers for configuring experiment logging."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure global logging handlers.

    Parameters
    ----------
    log_level: int
        Logging level for the root logger.
    log_file: Optional[Path]
        If provided, append a file handler pointing to this path.
    """
    logger = logging.getLogger()
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


__all__ = ["setup_logging"]
