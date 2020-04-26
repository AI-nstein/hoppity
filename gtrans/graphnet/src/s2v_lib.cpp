#include "s2v_lib.h"
#include "config.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
#include <cassert>

int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}

int n2n_construct(int num_nodes, int num_edges, int* node_degrees, int* edge_pairs, long long* idxes, Dtype* vals)
{
    int nnz = 0;    
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + num_edges;

	for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = 0; j < node_degrees[i]; ++j)
        {
            assert(edge_pairs[nnz * 2 + 1] == i);
            vals[nnz] = cfg::msg_average ? 1.0 / node_degrees[i] : 1.0;
            row_ptr[nnz] = i;
            col_ptr[nnz] = edge_pairs[nnz * 2];
            nnz++;
        }
    }
    assert(nnz == num_edges);
    return 0;
}

int prepare_indices(const int num_graphs,
                    const int *num_nodes,
                    const int *num_edges,
                    void **list_of_edge_pairs, 
                    long long* edge_to_idx,
                    long long* edge_from_idx,
                    long long* g_idx)
{
    int offset = 0;
    int cur_edge = 0;    
    for (int i = 0; i < num_graphs; ++i)
    {
        int *edge_pairs = static_cast<int *>(list_of_edge_pairs[i]);
        for (int j = 0; j < num_edges[i] * 2; j += 2)
        {
            int x = offset + edge_pairs[j];
            int y = offset + edge_pairs[j + 1];
            edge_to_idx[cur_edge] = y;
            edge_from_idx[cur_edge] = x;
            cur_edge += 1;
        }
        for (int j = 0; j < num_nodes[i]; ++j)
            g_idx[offset + j] = i;
        offset += num_nodes[i];
    }
    return 0;
}
