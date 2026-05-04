/*
 * Fast Dijkstra implementation for 2D non-uniform grids.
 * Supports 8, 16, or 32-connected neighborhoods.
 * 
 * Compile with pybind11:
 *   c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
 *       dijkstra_cpp.cpp -o dijkstra_cpp$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

// 8-connected neighborhood offsets (king's moves)
constexpr int DZ_8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
constexpr int DR_8[8] = {0, 0, -1, 1, -1, 1, -1, 1};

// 16-connected: 8 + knight's moves (2,1)
constexpr int DZ_16[16] = {-1, 1, 0, 0, -1, -1, 1, 1,
                           -2, -2, 2, 2, -1, 1, -1, 1};
constexpr int DR_16[16] = {0, 0, -1, 1, -1, 1, -1, 1,
                           -1, 1, -1, 1, -2, -2, 2, 2};

// 32-connected: 16 + extended knight's moves (3,1), (3,2)
constexpr int DZ_32[32] = {-1, 1, 0, 0, -1, -1, 1, 1,
                           -2, -2, 2, 2, -1, 1, -1, 1,
                           -3, -3, 3, 3, -1, 1, -1, 1,
                           -3, -3, 3, 3, -2, 2, -2, 2};
constexpr int DR_32[32] = {0, 0, -1, 1, -1, 1, -1, 1,
                           -1, 1, -1, 1, -2, -2, 2, 2,
                           -1, 1, -1, 1, -3, -3, 3, 3,
                           -2, 2, -2, 2, -3, -3, 3, 3};

py::array_t<double> arrival_time_dijkstra_cpp(
    py::array_t<double> r_um,
    py::array_t<double> z_um,
    py::array_t<double> etch_rate_map,
    int connectivity = 8
) {
    // Get buffer info
    auto r_buf = r_um.request();
    auto z_buf = z_um.request();
    auto etch_buf = etch_rate_map.request();
    
    const double* r_ptr = static_cast<double*>(r_buf.ptr);
    const double* z_ptr = static_cast<double*>(z_buf.ptr);
    const double* etch_ptr = static_cast<double*>(etch_buf.ptr);
    
    const size_t n_r = r_buf.shape[0];
    const size_t n_z = z_buf.shape[0];
    const size_t n_nodes = n_z * n_r;
    
    // Select neighbor offsets based on connectivity
    const int* DZ;
    const int* DR;
    int n_neighbors;
    
    if (connectivity == 32) {
        DZ = DZ_32;
        DR = DR_32;
        n_neighbors = 32;
    } else if (connectivity == 16) {
        DZ = DZ_16;
        DR = DR_16;
        n_neighbors = 16;
    } else {
        DZ = DZ_8;
        DR = DR_8;
        n_neighbors = 8;
    }
    
    // Output array
    py::array_t<double> result({n_z, n_r});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize distances to infinity
    constexpr double INF = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n_nodes; ++i) {
        result_ptr[i] = INF;
    }
    
    // Priority queue: (distance, node_index)
    using PQElement = std::pair<double, size_t>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    
    // Find z=0 row and initialize all starting nodes
    size_t iz_start = 0;
    double min_z_abs = std::abs(z_ptr[0]);
    for (size_t iz = 1; iz < n_z; ++iz) {
        double z_abs = std::abs(z_ptr[iz]);
        if (z_abs < min_z_abs) {
            min_z_abs = z_abs;
            iz_start = iz;
        }
    }
    
    // Add all starting nodes (entire row at z=0)
    for (size_t ir = 0; ir < n_r; ++ir) {
        size_t node = iz_start * n_r + ir;
        result_ptr[node] = 0.0;
        pq.push({0.0, node});
    }
    
    // Dijkstra main loop
    while (!pq.empty()) {
        auto [dist, node] = pq.top();
        pq.pop();
        
        // Skip if we've already found a shorter path
        if (dist > result_ptr[node]) continue;
        
        size_t iz = node / n_r;
        size_t ir = node % n_r;
        
        double z_curr = z_ptr[iz];
        double r_curr = r_ptr[ir];
        double speed_curr = etch_ptr[node];
        
        if (speed_curr <= 0) continue;
        
        // Check all neighbors
        for (int k = 0; k < n_neighbors; ++k) {
            int iz_nb = static_cast<int>(iz) + DZ[k];
            int ir_nb = static_cast<int>(ir) + DR[k];
            
            if (iz_nb < 0 || iz_nb >= static_cast<int>(n_z) ||
                ir_nb < 0 || ir_nb >= static_cast<int>(n_r)) {
                continue;
            }
            
            size_t neighbor = static_cast<size_t>(iz_nb) * n_r + static_cast<size_t>(ir_nb);
            double speed_nb = etch_ptr[neighbor];
            
            if (speed_nb <= 0) continue;
            
            double z_nb = z_ptr[iz_nb];
            double r_nb = r_ptr[ir_nb];
            double dz = z_nb - z_curr;
            double dr = r_nb - r_curr;
            double edge_dist = std::sqrt(dz * dz + dr * dr);
            
            // Harmonic mean speed
            double speed_avg = 2.0 * speed_curr * speed_nb / (speed_curr + speed_nb);
            double travel_time = edge_dist / speed_avg;
            
            double new_dist = dist + travel_time;
            
            if (new_dist < result_ptr[neighbor]) {
                result_ptr[neighbor] = new_dist;
                pq.push({new_dist, neighbor});
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(dijkstra_cpp, m) {
    m.doc() = "Fast C++ Dijkstra for 2D non-uniform grids with variable connectivity";
    m.def("arrival_time_dijkstra_cpp", &arrival_time_dijkstra_cpp,
          "Compute arrival times using Dijkstra on a 2D speed map",
          py::arg("r_um"), py::arg("z_um"), py::arg("etch_rate_map"),
          py::arg("connectivity") = 8);
}
