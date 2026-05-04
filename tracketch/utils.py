"""Package-level configuration helpers."""

import yaml
from pathlib import Path


def load_config(cfg_name: str, folder_name: str = "config") -> dict:
    """Load a YAML configuration file from the package ``config/`` directory.

    Parameters
    ----------
    cfg_name : str
        Filename, e.g. ``"cfg_libamtrack.yaml"``.
    folder_name : str
        Subdirectory inside the package root (default ``"config"``).

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    cfg_path = get_project_root() / folder_name / cfg_name

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. Expected in 'tracketch/config/*.yaml'."
        )

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Return the path to the ``tracketch/`` package directory."""
    return Path(__file__).parent
