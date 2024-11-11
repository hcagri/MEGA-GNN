#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <algorithm>
#include <random>

namespace py = pybind11;

std::vector<std::vector<int>> generate_negative_samples(const std::vector<std::vector<int>>& edge_index, 
                                                        const std::vector<std::vector<int>>& pos_edge_index, 
                                                        int num_neg_samples) {
    if (num_neg_samples <= 0) {
        throw std::invalid_argument("num_neg_samples must be greater than 0");
    }

    std::vector<int> neg_srcs;
    std::vector<int> neg_dsts;

    std::set<int> nodeset;
    std::unordered_map<int, std::set<int>> adj_list;

    for (size_t i = 0; i < edge_index[0].size(); ++i) {
        int src = edge_index[0][i];
        int dst = edge_index[1][i];
        nodeset.insert(src);
        nodeset.insert(dst);
        adj_list[src].insert(dst);
        adj_list[dst].insert(src);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, nodeset.size() - 1);

    int src, dst, neg_node;
    for(int i = 0; i < pos_edge_index[0].size(); i++){
        src = pos_edge_index[0][i];
        dst = pos_edge_index[1][i];
        std::set<int> unavail_nodes;
        unavail_nodes.insert(src);
        unavail_nodes.insert(dst);
        unavail_nodes.insert(adj_list[src].begin(), adj_list[src].end());
        unavail_nodes.insert(adj_list[dst].begin(), adj_list[dst].end());


        for(int j = 0; j < num_neg_samples/2; j++){
            do {
                neg_node = dis(gen);
            } while (unavail_nodes.count(neg_node) > 0);
            neg_srcs.push_back(src);
            neg_dsts.push_back(neg_node);
        }
        for(int j = 0; j < num_neg_samples/2; j++){
            do {
                neg_node = dis(gen);
            } while (unavail_nodes.count(neg_node) > 0);
            neg_srcs.push_back(neg_node);
            neg_dsts.push_back(dst);
        }
    }
    return {neg_srcs, neg_dsts};
}

PYBIND11_MODULE(negative_sampling, m) {
    m.def("generate_negative_samples", &generate_negative_samples, "A function to generate negative samples",
          py::arg("edge_index"), py::arg("pos_edge_index"), py::arg("num_neg_samples"));
}