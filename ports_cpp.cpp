#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <set>

namespace py = pybind11;

struct Triplet {
    int first;
    int second;
    int third;

    Triplet(int f, int s, int t) : first(f), second(s), third(t) {}
};

typedef std::unordered_map<int, std::vector<Triplet>> AdjList;

py::array_t<int> ports(py::array_t<int> edge_index, const AdjList& adj_list) {
    auto edge_index_r = edge_index.unchecked<2>();
    ssize_t M = edge_index_r.shape(1);

    // Initialize ports array
    py::array_t<int> ports_array(M);
    auto ports_r = ports_array.mutable_unchecked<1>();

    std::vector<Triplet> sorted_nbs;
    
    // Iterate over adjacency list and create port mappings
    for (const auto& [v, nbs] : adj_list) {
        if (nbs.empty()) continue;

        std::unordered_map<int, int> mapping;
        int counter = 0;
        sorted_nbs = nbs;
        
        // Sort neighbors by the second element (time)
        std::sort(sorted_nbs.begin(), sorted_nbs.end(), [](const Triplet& a, const Triplet& b) {
            return a.second < b.second;
        });

        for (const auto& [target, timestamp, idx] : sorted_nbs){
            if (mapping.find(target) != mapping.end()) {
                ports_r(idx) = mapping[target];
            } else {
                mapping[target] = counter;
                ports_r(idx) = counter;
                counter++;
            }
        }
        
    }
        
    return ports_array;
}

py::tuple assign_ports(py::array_t<int> arr, py::array_t<int> edge_index, int num_nodes) {
    AdjList adj_list_out;
    AdjList adj_list_in;
    int u, v, t;

    adj_list_in.reserve(num_nodes);
    adj_list_out.reserve(num_nodes);
    
    auto r = arr.unchecked<2>();  // Unchecked array access for speed
    for (int i = 0; i < r.shape(0); ++i) {
        u = r(i, 0);
        v = r(i, 1);
        t = r(i, 2);
        adj_list_out[u].emplace_back(v, t, i);
        adj_list_in[v].emplace_back(u, t, i);
    }

    py::array_t<int> ports_in = ports(edge_index, adj_list_in);
    py::array_t<int> ports_out = ports(edge_index, adj_list_out);
    
    return py::make_tuple(ports_in, ports_out);
}


PYBIND11_MODULE(ports_cpp, m) {
    m.doc() = "Convert edge list to adjacency lists with times";

    m.def("assign_ports", &assign_ports, "Convert edge list to adjacency lists with times",
          py::arg("arr"), py::arg("edge_index"), py::arg("num_nodes"));
}
