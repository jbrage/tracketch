# %%
"""
Using the tracketch logger.

tracketch uses Python's standard ``logging`` module.  By default the library
is silent (a NullHandler is registered at the package level).

The simplest way to activate output is to pass ``log_level`` directly to
:class:`~tracketch.TrackSimulator`.  For advanced use you can also configure
the ``"tracketch"`` logger yourself before instantiating anything.
"""

import logging
from tracketch import TrackSimulator

# -- Option 1: log_level parameter (recommended) ---------------------------
# Pass logging.DEBUG, logging.INFO, or a string such as "DEBUG".

sim_debug = TrackSimulator(
    particle_name="12C",
    start_energy_MeV_u=270.0,
    log_level=logging.DEBUG,
)

# %%
sim_info = TrackSimulator(
    particle_name="4He",
    start_energy_MeV_u=50.0,
    log_level=logging.INFO,
)

# %%
# -- Option 2: configure the logger manually (advanced) ---------------------
# Useful when you want a custom format, file handler, or need to share the
# handler with the rest of your application.

tracketch_logger = logging.getLogger("tracketch")
tracketch_logger.setLevel(logging.DEBUG)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
tracketch_logger.addHandler(_h)

sim_custom = TrackSimulator(
    particle_name="1H",
    start_energy_MeV_u=8.0,
)
