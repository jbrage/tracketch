Examples
========

The ``examples/`` directory contains runnable scripts that demonstrate the main
features of tracketch.  Each section below shows the code and explains what it
does.


Simulating track shapes
-----------------------

The most common use-case: create a :class:`~tracketch.TrackSimulator`, then
extract the etched track contour at several etching durations.

.. literalinclude:: ../examples/plot_track_shapes.py
   :language: python
   :start-after: # %%

The contour is returned as ``(r, z)`` arrays in um.  Mirroring ``-r`` gives
the full axially-symmetric profile.


Hydrogen isotopes -- proton, deuteron, triton
----------------------------------------------

.. versionadded:: 0.1

tracketch fully supports all three hydrogen isotopes (``"1H"``, ``"2H"``,
``"3H"``) as particle names.  The physics is correct for each isotope:

* **LET** is identical at the same kinetic energy per nucleon (MeV/u) -- it
  depends only on charge and velocity, not on mass number.
* **CSDA range** scales linearly with mass number *A*, because a heavier
  isotope carries more momentum at the same MeV/u.
* **Total kinetic energy** (MeV) = energy per nucleon x *A*.

The example below computes track radii for proton, deuteron, and triton across
a range of energies, illustrating the range difference at the same MeV/u:

.. literalinclude:: ../examples/plot_hydrogen_isotopes_radii.py
   :language: python
   :start-after: # %%


Radial dose distributions
-------------------------

The radial dose distribution (RDD) describes how dose falls off with distance
from the ion path.  tracketch wraps the Cucinotta model from libamtrack and
scales it to the target material.

.. literalinclude:: ../examples/plot_RDD.py
   :language: python
   :start-after: # %%


LET and CSDA range
------------------

Linear energy transfer (LET) and CSDA range can be queried from SRIM tables
or from libamtrack.  The plot includes hydrogen isotopes to illustrate that
LET curves for ``1H``, ``2H``, and ``3H`` coincide while CSDA ranges differ
by a factor of *A*.

.. literalinclude:: ../examples/plot_LET_CSDA.py
   :language: python
   :start-after: # %%


Customising the simulation grid
--------------------------------

By default :class:`~tracketch.TrackSimulator` uses a fixed radial and depth
grid.  For ions with very long tracks, very short tracks, or when a different
resolution is needed, individual grid parameters can be passed as plain keyword
arguments:

.. code-block:: python

   sim = TrackSimulator(
       particle_name="12C",
       start_energy_MeV_u=10.0,
       r_max_um=40,   # wider radial grid
       z_max_um=80,   # deeper depth grid
       n_points_z=200,
   )

The full example below runs the same ion three times (default, wide, and
coarse grids) and overlays the resulting iso-time contours:

.. literalinclude:: ../examples/plot_custom_grid.py
   :language: python
   :start-after: # %%


Comparing arrival-time solvers
------------------------------

tracketch ships two wavefront propagation backends: Fast Marching (FMM) and
Dijkstra.  This example compares them on a uniform-speed grid.

.. literalinclude:: ../examples/plot_arrival_times.py
   :language: python
   :start-after: # %%
