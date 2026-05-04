import pandas as pd
from pathlib import Path

from tracketch.etching.etch_rate_model_io import default_etch_rate_model
from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch import TrackSimulator
from tracketch.physics import convert_MeV_to_MeV_u

_DATA_DIR = Path(__file__).parent / "data"


def load_reference_data_track_shape(
    particle_names: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the Doerschel measured 2D track shapes (tracks perpendicular to the surface.)
    Filter the relevant particle names.
    Return the filtered DataFrame and a dictionary mapping particle names to their energies.
    """

    # load all data
    reference_df = pd.read_csv(_DATA_DIR / "doerschel-data.csv")
    reference_df.dropna(inplace=True)

    # get the relevant particles
    if particle_names is not None:
        filter_df = reference_df[reference_df.particle_name.isin(particle_names)]
    else:
        filter_df = reference_df

    # create a dict that maps the energies used for each particle
    particle_energy_MeV_dict = (
        filter_df.groupby("particle_name").Energy_MeV.unique().to_dict()
    )

    # convert the energies to floats (from list)
    for key, value in particle_energy_MeV_dict.items():
        particle_energy_MeV_dict[key] = float(value[0])

    return filter_df, particle_energy_MeV_dict


def load_reference_data_track_length(
    particle_names: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the Doerschel data measured track lengths as a function of etching time and particle type.
    Filter the relevant particle names.
    Return the filtered DataFrame and a dictionary mapping particle names to their energies.
    """

    # load all data
    reference_df = pd.read_csv(_DATA_DIR / "track_length_df.csv")
    reference_df.dropna(inplace=True)

    # reference_df = reference_df[
    #     reference_df.Energy_MeV > 20
    #     ]

    # get the relevant particles
    if particle_names is not None:
        filter_df = reference_df[reference_df.particle_name.isin(particle_names)]
    else:
        filter_df = reference_df

    # create a dict that maps the energies used for each particle
    particle_energy_MeV_dict = (
        filter_df.groupby("particle_name").Energy_MeV.unique().to_dict()
    )

    return filter_df, particle_energy_MeV_dict


def load_reference_data_angles(
    particle_names: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the Doerschel data for tracks at various angles.
    Filter the relevant particle names.
    Return the filtered DataFrame and a dictionary mapping particle names to their energies.
    """

    # load all data
    reference_df = pd.read_csv(_DATA_DIR / "doerschel-angle.csv")
    reference_df.dropna(inplace=True)

    # get the relevant particles
    if particle_names is not None:
        filter_df = reference_df[reference_df.particle_name.isin(particle_names)]
    else:
        filter_df = reference_df

    # create a dict that maps the energies used for each particle
    particle_energy_MeV_dict = (
        filter_df.groupby("particle_name").Energy_MeV.unique().to_dict()
    )

    # convert the energies to floats (from list)
    for key, value in particle_energy_MeV_dict.items():
        particle_energy_MeV_dict[key] = float(value[0])

    return filter_df, particle_energy_MeV_dict


def create_minimisation_data_track_shape(
    particle_names: list[str] | None = None,
    etch_model: EtchRateModel = default_etch_rate_model(),
    arrival_time_method: str = "dijkstra",
) -> dict[str, dict]:
    """
    For data: 2D track shapes (tracks perpendicular to the surface.)

    Create a data structure for minimisation that contains the dose maps, etch models etc for the given reference data and particle energies.
    Returns a dictionary mapping particle names to their simulator objects and experimental data.
    The dose maps are time consuming to create, so we create them once here.
    """

    # load all track shape data
    reference_df, particle_energy_MeV_dict = load_reference_data_track_shape(
        particle_names=particle_names
    )

    # create a dict that stores the dose maps and experimental data for each particle
    detector_dict = {}
    print("Creating minimisation data for track shape ..")
    for particle_name in particle_energy_MeV_dict.keys():
        energy_MeV = particle_energy_MeV_dict[particle_name]

        print(f"\t{particle_name} with energy {energy_MeV} MeV")

        # convert the energy to MeV/u
        energy_MeV_u = convert_MeV_to_MeV_u(
            particle_name=particle_name, Energy_MeV=energy_MeV
        )

        # look up the experimental data for this particle
        particle_df = reference_df[reference_df.particle_name == particle_name]

        # create the TrackSimulator object with the etch model and rz limits that approx spans the exp data
        sim = TrackSimulator(
            particle_name=particle_name,
            start_energy_MeV_u=energy_MeV_u,
            etch_model=etch_model,
            arrival_time_method_name=arrival_time_method,
            rz_lims_dict={
                "z_max_um": particle_df.z_um.max() * 2.0,
                "r_max_um": particle_df.r_um.max() * 2.0,
                "n_points_z": 200,
                "n_points_r": 200,
            },
        )

        # store in the dict
        detector_dict[particle_name] = {
            "simulator": sim,
            "experiment_data": particle_df,
            "etching_times_h": particle_df.time_h.unique(),
        }
    return detector_dict


def create_minimisation_data_track_length(
    particle_names: list[str] | None = None,
    etch_model: EtchRateModel = default_etch_rate_model(),
    arrival_time_method: str = "dijkstra",
) -> dict[str, dict]:
    """
    For data: track lengths as a function of etching time and particle type.

    Create a data structure for minimisation that contains the dose maps, etch models etc for the given reference data and particle energies.
    Returns a dictionary mapping particle names to their simulator objects and experimental data.
    The dose maps are time consuming to create, so we create them once here.
    """

    # load all track length data
    reference_df, particle_energy_MeV_dict = load_reference_data_track_length(
        particle_names=particle_names
    )

    z_buffer_um = 25.0  # add some buffer to the z limits to ensure we capture the full track length

    # create a dict that stores the dose maps and experimental data for each particle
    detector_dict = {}
    print("Creating minimisation data for track length ..")
    for particle_name in particle_energy_MeV_dict.keys():
        energies_MeV = particle_energy_MeV_dict[particle_name]
        for energy_MeV in energies_MeV:
            print(f"\t{particle_name} with energy {energy_MeV} MeV")

            # convert the energy to MeV/u
            energy_MeV_u = convert_MeV_to_MeV_u(
                particle_name=particle_name, Energy_MeV=energy_MeV
            )

            # look up the experimental data for this particle-energy pair
            particle_df = reference_df[
                (reference_df.particle_name == particle_name)
                & (reference_df.Energy_MeV == energy_MeV)
            ]

            if particle_df.empty:
                continue

            # create the TrackSimulator object with the etch model and rz limits that approx spans the exp data
            sim = TrackSimulator(
                particle_name=particle_name,
                start_energy_MeV_u=energy_MeV_u,
                etch_model=etch_model,
                arrival_time_method_name=arrival_time_method,
                rz_lims_dict={
                    "z_max_um": particle_df.length_um.max() + z_buffer_um,
                    "r_max_um": 10.0,  # not relevant for track length data
                    "n_points_z": 200,
                    "n_points_r": 200,
                },
            )

            # store in the dict
            key = f"{particle_name}_{energy_MeV:g}MeV"
            detector_dict[key] = {
                "simulator": sim,
                "etching_times_h": particle_df.time_h.unique(),
                "experiment_data": particle_df,
            }
    return detector_dict


def create_minimisation_data(
    particle_names: list[str] | None = None,
    etch_model: EtchRateModel = default_etch_rate_model(),
    arrival_time_method: str = "dijkstra",
) -> dict[str, dict[str, dict]]:
    """
    Create combined minimisation datasets for shape and length optimization.
    Returns a dictionary with keys "track_shape" and "track_length", each containing a dict mapping particle names to their simulator objects and experimental data.
    """
    models_dict_shape = create_minimisation_data_track_shape(
        particle_names=particle_names,
        etch_model=etch_model,
        arrival_time_method=arrival_time_method,
    )
    models_dict_length = create_minimisation_data_track_length(
        particle_names=particle_names,
        etch_model=etch_model,
        arrival_time_method=arrival_time_method,
    )
    return {
        "track_shape": models_dict_shape,
        "track_length": models_dict_length,
    }


if __name__ == "__main__":
    df, particles_dict = load_reference_data_track_shape()
    df, particles_dict = load_reference_data_track_length()

    print(df.head())
