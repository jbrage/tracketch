Algorithms
==========

This page gives a brief overview of the computational methods used inside
tracketch.  For full API details see :doc:`api/index`.


Simulation pipeline
-------------------

A :class:`~tracketch.TrackSimulator` runs three stages:

1. **Dose map** -- For each depth step along the ion path, the *radial dose
   distribution* (RDD) is evaluated using a model from libamtrack (e.g.
   Cucinotta).  SRIM stopping-power tables provide LET and residual energy at
   each depth, so the RDD changes as the ion slows down.

2. **Etch-rate map** -- The calibrated :class:`~tracketch.EtchRateModel`
   converts local dose *D* to a local etch velocity *V(D)* via monotonic PCHIP
   interpolation in log-dose space.  Undamaged material etches at the constant
   bulk rate *V_bulk*.

3. **Arrival-time map** -- Starting from the detector surface (*z* = 0), the
   etchant front is propagated through the spatially varying speed field.
   This is a shortest-path / wavefront problem solved by one of the backends
   below.


Arrival-time solvers
--------------------

The etch-rate map defines a *speed* at every grid point.  The arrival time
at each cell is the minimum travel time from the surface through any path,
weighted by inverse speed:

.. math::

   T(\mathbf{x}) = \min_{\text{paths}} \int_0^{\mathbf{x}}
   \frac{\mathrm{d}s}{V(s)}

tracketch provides two solver families.


Dijkstra (recommended)
~~~~~~~~~~~~~~~~~~~~~~

The 2-D grid is treated as a weighted graph.  Each cell is a node connected to
its neighbours; edge weights are the physical travel time across the cell
boundary:

.. math::

   w_{ij} = \frac{|\mathbf{x}_i - \mathbf{x}_j|}
   {\tfrac{1}{2}(V_i + V_j)}

Because the radial grid is log-spaced (necessary to resolve narrow RDDs at
small *r*), cell sizes vary by orders of magnitude.  Dijkstra handles
non-uniform spacing natively -- no regridding required.

**Connectivity.**
Higher connectivity reduces *metrication error* (the angular bias from a
discrete grid):

- **8-connected** (default) -- king's moves; ~2 % error; fastest.
- **16-connected** -- adds knight's moves ``(+/-2, +/-1)``; ~1 % error.
- **32-connected** -- extended knight's moves; ~0.5 % error.

**Backends.**
The graph is built with Numba JIT and then solved with
``scipy.sparse.csgraph.dijkstra``:

- ``dijkstra_numba`` -- pure Python + Numba; always available.
- ``dijkstra_cpp`` -- C++/pybind11 implementation of the graph builder;
  ~50-100x faster.  Requires compilation (see :doc:`getting_started`).


Fast Marching Method (FMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The `scikit-fmm <https://github.com/scikit-fmm/scikit-fmm>`_ library solves
the Eikonal equation on a *uniform* grid.  Since the native radial grid is
log-spaced, tracketch interpolates the speed map onto a uniform grid, runs
FMM, then interpolates back.

FMM supports **tilted tracks** (``theta_deg > 0``), where the wavefront
starts along a diagonal rather than a flat surface.  Dijkstra does not yet
support tilting.


Etch-rate model
---------------

:class:`~tracketch.EtchRateModel` represents *V(D)* as a monotonic PCHIP
(Piecewise Cubic Hermite Interpolating Polynomial) spline through a set of
dose-velocity anchor points.  Key design choices:

- Anchor velocities are stored as *cumulative increments* to enforce
  monotonicity by construction.
- A synthetic low-dose anchor at 10\ :sup:`-3` Gy is prepended so that the
  spline smoothly converges to *V_bulk* at low doses.
- Extrapolation beyond the highest anchor uses the last-segment slope (PCHIP
  mode) or clamps to the last value.

The model can be serialised to JSON and shared across simulations.
