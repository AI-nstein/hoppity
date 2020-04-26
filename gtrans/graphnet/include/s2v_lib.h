#ifndef S2V_LIB_H
#define S2V_LIB_H

#include "config.h"

extern "C" int Init(const int argc, const char **argv);

extern "C" int n2n_construct(int num_nodes, int num_edges, int* node_degrees, int* edge_pairs, long long* idxes, Dtype* vals);

extern "C" int prepare_indices(const int num_graphs, const int *num_nodes, const int *num_edges, void **list_of_edge_pairs, long long* edge_to_idx, long long* edge_from_idx, long long* g_idx);

#endif