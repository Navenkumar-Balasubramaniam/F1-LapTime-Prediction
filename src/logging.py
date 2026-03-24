"""
Logging utilities for the MLOps pipeline.

Why this module exists:
- Centralize logging setup so every module uses the same format and log levels.
- Write logs both to the console and to a log file for debugging and experiment traceability.
- Keep logging configuration separate from training and orchestration logic.

This file intentionally uses the standard library logging module so the project
stays simple and does not depend on any heavy logging framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_file: Path | str = Path("logs/pipeline.log"), level: str = "INFO") -> Path:
    """
    Configure application-wide logging.

    Parameters
    ----------
    log_file : Path | str
        File path where logs should be persisted.
    level : str
        Logging level name, for example: DEBUG, INFO, WARNING, ERROR.

    Returns
    -------
    Path
        Resolved path to the log file.

    Notes
    -----
    - Existing handlers are cleared so repeated test runs do not duplicate logs.
    - We log to both stdout and a file to support local debugging and artifact upload.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear handlers so repeated calls remain deterministic in notebooks/tests.
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.getLogger(__name__).info("Logging configured. Writing logs to %s", log_path)
    return log_path


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a module logger.

    Parameters
    ----------
    name : Optional[str]
        Usually __name__ from the calling module.
    """
    return logging.getLogger(name)
