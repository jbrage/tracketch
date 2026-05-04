"""
C++ Dijkstra wrapper - optional high-performance backend.

To use, compile the extension first:
    cd tracketch/wavefront/dijkstra/cpp
    python setup_dijkstra.py build_ext --inplace
"""

try:
    from .dijkstra_cpp_wrapper import arrival_time_dijkstra_cpp

    __all__ = ["arrival_time_dijkstra_cpp"]
except ImportError:
    __all__ = []
