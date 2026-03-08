"""
src/utils/paths.py
------------------
Centralized path definitions for the BRFSS Obesity Early Warning project.

All source modules should import from here rather than hard-coding paths.
This ensures paths are correct regardless of which directory the script is
invoked from.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — parent of the directory containing this file
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
SRC_DIR: Path = PROJECT_ROOT / "src"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------------
# Data subdirectories
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Model subdirectories
# ---------------------------------------------------------------------------
REGRESSION_MODELS_DIR: Path = MODELS_DIR / "regression"
CLASSIFICATION_MODELS_DIR: Path = MODELS_DIR / "classification"

# ---------------------------------------------------------------------------
# Report subdirectories
# ---------------------------------------------------------------------------
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------
CONFIG_FILE: Path = CONFIGS_DIR / "config.yaml"


def ensure_dirs() -> None:
    """Create all output directories if they do not already exist.

    Call this at pipeline startup to guarantee all write targets exist.
    """
    dirs_to_create = [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REGRESSION_MODELS_DIR,
        CLASSIFICATION_MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)


def raw_data_path(filename: str) -> Path:
    """Return the full path to a file in data/raw/.

    Args:
        filename: Name of the raw data file.

    Returns:
        Absolute path to the raw data file.
    """
    return RAW_DATA_DIR / filename


def interim_data_path(filename: str) -> Path:
    """Return the full path to a file in data/interim/."""
    return INTERIM_DATA_DIR / filename


def processed_data_path(filename: str) -> Path:
    """Return the full path to a file in data/processed/."""
    return PROCESSED_DATA_DIR / filename


def model_path(track: str, filename: str) -> Path:
    """Return the full path to a serialized model file.

    Args:
        track: Either 'regression' or 'classification'.
        filename: Model filename (e.g., 'random_forest.joblib').

    Returns:
        Absolute path to the model file.
    """
    if track == "regression":
        return REGRESSION_MODELS_DIR / filename
    elif track == "classification":
        return CLASSIFICATION_MODELS_DIR / filename
    else:
        raise ValueError(f"Unknown model track: '{track}'. Expected 'regression' or 'classification'.")


def metrics_path(filename: str) -> Path:
    """Return the full path to a metrics JSON file in reports/metrics/."""
    return METRICS_DIR / filename


def figures_path(filename: str) -> Path:
    """Return the full path to a figure in reports/figures/."""
    return FIGURES_DIR / filename


if __name__ == "__main__":
    # Quick sanity check when run directly
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"Reports dir  : {REPORTS_DIR}")
    print(f"Config file  : {CONFIG_FILE}")
