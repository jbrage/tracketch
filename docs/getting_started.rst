Getting Started
===============

Installation
------------

Clone the repository and install in a virtual environment:

.. code-block:: bash

   git clone <repo-url> tracketch
   cd tracketch
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[numba]"

This installs tracketch with the Numba-accelerated Dijkstra backend, which
works on any Linux machine.

.. note::

   tracketch depends on `libamtrack <https://libamtrack.github.io/>`_ and
   is currently **Linux only**.

.. note::

   This release currently supports only 2D axisymmetric simulations
   (straight tracks).

Optional: C++ Dijkstra backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ~100x faster arrival-time computation:

.. code-block:: bash

   pip install -e ".[numba,cpp]"
   cd tracketch/wavefront/dijkstra/cpp
   python setup_dijkstra.py build_ext --inplace


Quick example
-------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from tracketch import TrackSimulator

   # 270 MeV/u carbon-12 hitting CR-39
   sim = TrackSimulator(
       particle_name="12C",
       start_energy_MeV_u=270.0,
   )

   # Plot track shape after 1 and 3 hours of etching
   fig, ax = plt.subplots()
   for t in [1.0, 3.0]:
       r, z = sim.get_iso_time_contour(etching_time_h=t)
       ax.plot(r, z, label=f"{t} h")
       ax.plot(-r, z, color=ax.lines[-1].get_color())

   ax.set_xlabel("r / um")
   ax.set_ylabel("z / um")
   ax.invert_yaxis()
   ax.legend(title="Etch time")
   ax.set_aspect("equal")
   plt.tight_layout()
   plt.show()

The :class:`~tracketch.TrackSimulator` automatically computes the dose map,
etch-rate map, and arrival-time map on construction.  See :doc:`examples` for
more detailed scripts.


Etch-rate model selection
-------------------------

tracketch uses one shared default etch-rate calibration model,
``"Doerschel_etching"``, for all particle species.

- The etch-rate model is **not** selected from ``particle_name``.
- ``particle_name`` only controls stopping power / dose-map physics.
- You can provide a custom etch model explicitly when needed.

.. code-block:: python

   from tracketch import TrackSimulator, load_etchrate_model

   etch_model = load_etchrate_model("my_custom_model")

   sim = TrackSimulator(
      particle_name="12C",
      start_energy_MeV_u=270.0,
      etch_model=etch_model,
   )


Particles and stopping-power sources
-------------------------------------

Particle names follow the ``"<A><symbol>"`` convention, where *A* is the mass
number and *symbol* is the element symbol -- for example ``"12C"``, ``"1H"``,
``"56Fe"``.

The stopping-power source is chosen with the ``stopping_power_source``
parameter:

``"SRIM"`` (default)
   Uses tabulated SRIM data.  Only the following ten particles are supported:

   .. code-block:: python

      import tracketch
      print(tracketch.SRIM_PARTICLES)
      # ('1H', '2H', '3H', '4He', '7Li', '9Be', '11B', '12C', '14N', '16O')

   Requesting any other particle raises a :class:`ValueError` with a hint to
   switch to ``"libamtrack"``.

   .. note::

      All three hydrogen isotopes are supported: ``"1H"`` (proton), ``"2H"``
      (deuteron), ``"3H"`` (triton).  The CSDA range scales linearly with mass
      number *A*; LET at the same MeV/u is identical across isotopes.

``"libamtrack"``
   Uses the PSTAR/ASTAR parametrisation via
   `libamtrack <https://libamtrack.github.io/>`_.  Accepts any nuclide
   recognised by libamtrack, including heavy ions such as ``"56Fe"`` or
   ``"238U"``.

.. code-block:: python

   from tracketch import TrackSimulator

   # Heavy ion -- switch to libamtrack for stopping power
   sim = TrackSimulator(
       particle_name="56Fe",
       start_energy_MeV_u=1000.0,
       stopping_power_source="libamtrack",
   )


Target material
---------------

Two materials are supported:

.. code-block:: python

   import tracketch
   print(tracketch.MATERIALS)   # ('CR39', 'water')

Pass the name as ``material_name="CR39"`` (default, density 1.31 g/cm^3) or
``material_name="water"`` (density 1.00 g/cm^3).  An unknown name raises a
:class:`ValueError`.


Simulation grid
---------------

The simulation uses a 2-D cylindrical grid in (*r*, *z*).  The radial axis is
log-spaced (to resolve the narrow dose core near *r* = 0); the depth axis is
uniform.  Default values:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``r_min_um``
     - ``1e-4`` um
     - Inner radial boundary (start of log-spaced axis)
   * - ``r_max_um``
     - ``20`` um
     - Outer radial boundary
   * - ``z_max_um``
     - ``40`` um
     - Maximum depth along the ion path
   * - ``n_points_r``
     - ``400``
     - Number of radial grid points
   * - ``n_points_z``
     - ``100``
     - Number of depth grid points

All five parameters can be passed directly to :class:`~tracketch.TrackSimulator`:

.. code-block:: python

   from tracketch import TrackSimulator

   sim = TrackSimulator(
       particle_name="12C",
       start_energy_MeV_u=270.0,
       r_max_um=40,     # wider radial window
       z_max_um=80,     # deeper track
       n_points_r=600,  # finer radial resolution
       n_points_z=200,  # finer depth resolution
   )

Increasing resolution improves accuracy at the cost of compute time.  For
publication-quality results consider doubling both ``n_points_r`` and
``n_points_z``.  For quick exploration the defaults are a good starting point.

.. note::

   Enabling parallel dose-map computation with ``n_jobs=-1`` reduces wall time
   for large grids (>=50 000 total cells) or when using
   ``stopping_power_source="libamtrack"``.  For the default 400 x 100 grid
   the inter-process overhead outweighs the gain; leave ``n_jobs=1``
   (the default) in that case.


Building the documentation
--------------------------

.. code-block:: bash

   cd docs
   make html

Then open ``_build/html/index.html`` in a browser.
