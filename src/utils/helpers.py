"""
src/utils/helpers.py
--------------------
Shared helper utilities for the BRFSS Obesity Early Warning project.

Provides:
  - Logging setup (consistent format across all modules)
  - YAML config loading
  - JSON metrics I/O
  - Dataframe I/O (CSV read/write with logging)
  - Timing decorator
"""

import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import yaml

from src.utils.paths import CONFIG_FILE


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger with a consistent format.

    Args:
        name: Logger name — typically use ``__name__`` from the calling module.
        level: Logging level. Default is INFO.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Path to the YAML config file. Defaults to
            :data:`src.utils.paths.CONFIG_FILE`.

    Returns:
        Parsed config as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = config_path or CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV and log the result.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns:
        Loaded :class:`pandas.DataFrame`.
    """
    logger = get_logger(__name__)
    logger.info(f"Reading CSV: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def write_csv(df: pd.DataFrame, path: Path, index: bool = False, **kwargs) -> None:
    """Write a DataFrame to CSV and log the result.

    Args:
        df: DataFrame to write.
        path: Destination path. Parent directories are created if needed.
        index: Whether to write the row index. Default is False.
        **kwargs: Additional keyword arguments forwarded to :meth:`DataFrame.to_csv`.
    """
    logger = get_logger(__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    logger.info(f"Wrote {len(df):,} rows to {path}")


# ---------------------------------------------------------------------------
# JSON Metrics I/O
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save a metrics dictionary to a JSON file.

    Args:
        metrics: Flat or nested dictionary of metric name → value.
        path: Destination file path. Parent directories are created if needed.
    """
    logger = get_logger(__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2, default=_json_serializable)
    logger.info(f"Saved metrics to {path}")


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load a metrics JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed metrics dictionary.
    """
    with path.open("r") as f:
        return json.load(f)


def _json_serializable(obj: Any) -> Any:
    """Convert non-serializable types (e.g., numpy floats) for JSON dumping."""
    import numpy as np  # local import to avoid hard dep at module level

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Timing Decorator
# ---------------------------------------------------------------------------

def timeit(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function.

    Usage::

        @timeit
        def my_function():
            ...
    """
    logger = get_logger(__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.2f}s")
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def set_pandas_display() -> None:
    """Set pandas display options for readable console output."""
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header to stdout.

    Args:
        title: Section title string.
        width: Total line width. Default is 70.
    """
    border = "=" * width
    padding = " " * max(0, (width - len(title) - 2) // 2)
    print(f"\n{border}")
    print(f"|{padding}{title}{padding}|")
    print(f"{border}\n")
