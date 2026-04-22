import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

LIB_NAME = 'liberata_metrics'


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a library logger.

    Usage (inside library modules):
        logger = get_logger(__name__)
        logger.debug("message")

    This logger installs a NullHandler by default so importing the library does not
    configure application logging or emit output. Application code (scripts/CLI)
    should call configure_logging() once at startup to enable handlers/formatting.

    Parameters
    ----------
    name : Optional[str]
        Optional module name to append to the library logger root. If provided the
        returned logger name will be "liberata_metrics.<name>".
    Returns
    -------
    logging.Logger
        Configured logger object (may carry NullHandler by default).
    """
    logger_name = f"{LIB_NAME}.{name}" if name else LIB_NAME
    logger = logging.getLogger(logger_name)
    
    # ensure importing the library doesn't add handlers or configure root
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger



def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    fmt: Optional[str] = None,
) -> None:
    """
    Configure application-level logging for scripts/CLIs.

    Call this once from top-level entrypoints (scripts, CLI) before performing work.

    Parameters
    ----------
    level : Optional[str]
        Logging level name (e.g., "DEBUG", "INFO"). If None, reads from environment
        variable LOG_LEVEL or defaults to "INFO".
    log_file : Optional[str]
        Path to a rotating log file. If provided, a RotatingFileHandler is added.
        Parent directories are created if necessary.
    max_bytes : int
        Maximum bytes per log file before rotation.
    backup_count : int
        Number of rotated log files to keep.
    fmt : Optional[str]
        Optional log format string. If None, a sensible default is used.

    Notes
    -----
    - This function configures the root logger. Library modules should not call this;
      only application code (scripts or CLI entrypoints) should configure logging.
    - Multiple calls are safe: if the root logger already has handlers, the function
      sets the level and returns early to avoid duplicate handlers in interactive runs.
    Example
    -------
    from liberata_metrics.logging import configure_logging
    configure_logging(level="DEBUG", log_file="logs/run.log")
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    
    # If handlers already configured (e.g., in REPL/pytest), set level and return
    if root.handlers:
        root.setLevel(numeric_level)
        return

    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    root.setLevel(numeric_level)