"""Etch-rate model persistence and default-model factory.

Provides :func:`save_etchrate_model`, :func:`load_etchrate_model`, and a
convenience factory :func:`default_etch_rate_model`.
"""

import logging

import numpy as np
from pathlib import Path
from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch.utils import load_config, get_project_root

logger = logging.getLogger(__name__)


def default_etch_rate_model(
    n_anchors: int = 15, bulk_rate_um_h: float = 1.73
) -> EtchRateModel:
    """Generate a generic etch-rate model initialised from Doerschel data.

    Parameters
    ----------
    n_anchors : int
        Number of fine-resolution anchor points in the high-dose region.
    bulk_rate_um_h : float
        Bulk etch rate in um/hr.

    Returns
    -------
    tracketch.etching.etch_rate_model.EtchRateModel
    """
    # defaults for interpolation
    etch_model = load_etchrate_model("Doerschel_etching")
    anchor_doses_Gy = etch_model.anchor_doses_Gy
    anchor_velocities_um_h = etch_model.anchor_velocities_um_h

    logspace_anchors_coarse = np.logspace(2, 8, num=10)
    # create new anchors, lowest proton doses 1e-2, highest carbon ~1e9 Gy
    logspace_anchors_fine = np.logspace(6.5, 9.5, num=n_anchors)
    # Combine, sort, and deduplicate to ensure strict monotonicity
    new_anchors = np.unique(
        np.sort(np.concatenate([logspace_anchors_coarse, logspace_anchors_fine]))
    )
    new_velocities = np.interp(new_anchors, anchor_doses_Gy, anchor_velocities_um_h)

    # set all velocities below 1e6 Gy to the bulk etch rate
    new_velocities[new_anchors < 1e6] = bulk_rate_um_h

    gen_etchrate_model = EtchRateModel(
        anchor_doses_Gy=new_anchors,
        anchor_velocities_um_h=new_velocities,
        V_bulk_um_h=bulk_rate_um_h,
        name="general_model",
    )
    return gen_etchrate_model


def get_model_path(model_name: str) -> Path:
    """
    Get the full path to a model file based on config.

    Parameters
    ----------
    model_name : str
        The model identifier (e.g., 'Doerschel_etching')
    Returns
    -------
    Path
        Full path to the model file

    Examples
    --------
    >>> path = get_model_path('Doerschel_etching')
    """
    cfg = load_config("cfg_etching_models.yaml")

    base_dir = cfg["model_paths"]["base_dir"]
    default_models = cfg["model_paths"]["default_models"]

    if model_name in default_models:
        filename = default_models[model_name]
    else:
        # Allow arbitrary user-defined model names without editing config.
        filename = f"{model_name}.json"

    model_path = (get_project_root() / base_dir / filename).resolve()
    return model_path


def save_etchrate_model(model: "EtchRateModel", model_name: str) -> None:
    """
    Save a model using the configured path.

    Parameters
    ----------
    model : tracketch.etching.etch_rate_model.EtchRateModel
        The model to save
    model_name : str
        The model identifier (e.g., 'Doerschel_etching')

    Examples
    --------
    >>> save_etchrate_model(my_model, 'my_custom_model')
    """
    path = get_model_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_to_json(str(path))
    logger.info("Saved etch model '%s' to %s", model_name, path)


def load_etchrate_model(model_name: str) -> "EtchRateModel":
    """
    Load a model using the configured path.

    Parameters
    ----------
    model_name : str
        The model identifier (e.g., 'Doerschel_etching')

    Returns
    -------
    tracketch.etching.etch_rate_model.EtchRateModel
        The loaded model

    Examples
    --------
    >>> model = load_etchrate_model('Doerschel_etching')
    """
    path = get_model_path(model_name)
    logger.info("Loading etch model '%s' from %s", model_name, path)
    return EtchRateModel.load_from_json(str(path))
