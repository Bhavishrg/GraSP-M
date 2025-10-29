#include <io/netmp.h>
#include <graphdb/offline_evaluator.h>
#include <graphdb/online_evaluator.h>
#include <utils/circuit.h>

#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <omp.h>

#include "utils.h"

using namespace graphdb;
using json = nlohmann::json;
namespace bpo = boost::program_options;

common::utils::Circuit<Ring> generateCircuit(int nP, int pid, size_t vec_size, size_t add_size) {

    std::cout << "Generating circuit" << std::endl;
    
    common::utils::Circuit<Ring> circ;

    // Distributed Graph
    size_t num_vert = 0.1 * vec_size;
    size_t num_edge = vec_size - num_vert;
    std::vector<size_t> subg_num_vert(nP);
    std::vector<size_t> subg_num_edge(nP);
    for (int i = 0; i < subg_num_vert.size(); ++i) {
        if (i != nP - 1) {
            subg_num_vert[i] = num_vert / nP;
            subg_num_edge[i] = num_edge / nP;
        } else {
            subg_num_vert[i] = num_vert / nP + num_vert % nP;
            subg_num_edge[i] = num_edge / nP + num_edge % nP;
        }
    }

    std::vector<std::vector<wire_t>> subg_vertex_list(nP);
    std::vector<std::vector<wire_t>> subg_vertex_pos_V(nP);
    std::vector<std::vector<wire_t>> subg_vertex_pos_S(nP);
    std::vector<std::vector<wire_t>> subg_vertex_pos_D(nP);
    for (int i = 0; i < subg_vertex_list.size(); ++i) {
        std::vector<wire_t> subg_vertex_list_party(subg_num_vert[i]);
        std::vector<wire_t> subg_vertex_pos_V_party(subg_num_vert[i]);
        std::vector<wire_t> subg_vertex_pos_S_party(subg_num_vert[i]);
        std::vector<wire_t> subg_vertex_pos_D_party(subg_num_vert[i]);
        for (int j = 0; j < subg_vertex_list_party.size(); ++j) {
            subg_vertex_list_party[j] = circ.newInputWire();
            subg_vertex_pos_V_party[j] = circ.newInputWire();
            subg_vertex_pos_S_party[j] = circ.newInputWire();
            subg_vertex_pos_D_party[j] = circ.newInputWire();
        }
        subg_vertex_list[i] = subg_vertex_list_party;
        subg_vertex_pos_V[i] = subg_vertex_pos_V_party;
        subg_vertex_pos_S[i] = subg_vertex_pos_S_party;
        subg_vertex_pos_D[i] = subg_vertex_pos_D_party;
    }

    std::vector<std::vector<wire_t>> subg_edge_list(nP);
    std::vector<std::vector<wire_t>> subg_edge_pos_V(nP);
    std::vector<std::vector<wire_t>> subg_edge_pos_S(nP);
    std::vector<std::vector<wire_t>> subg_edge_pos_D(nP);
    for (int i = 0; i < subg_edge_list.size(); ++i) {
        std::vector<wire_t> subg_edge_list_party(subg_num_edge[i]);
        std::vector<wire_t> subg_edge_pos_V_party(subg_num_edge[i]);
        std::vector<wire_t> subg_edge_pos_S_party(subg_num_edge[i]);
        std::vector<wire_t> subg_edge_pos_D_party(subg_num_edge[i]);
        for (int j = 0; j < subg_edge_list_party.size(); ++j) {
            subg_edge_list_party[j] = circ.newInputWire();
            subg_edge_pos_V_party[j] = circ.newInputWire();
            subg_edge_pos_S_party[j] = circ.newInputWire();
            subg_edge_pos_D_party[j] = circ.newInputWire();
        }
        subg_edge_list[i] = subg_edge_list_party;
        subg_edge_pos_V[i] = subg_edge_pos_V_party;
        subg_edge_pos_S[i] = subg_edge_pos_S_party;
        subg_edge_pos_D[i] = subg_edge_pos_D_party;
    }

    // Initialize vectors of edges to be added.
    // Distribute add_size across parties 
    std::vector<size_t> add_num_edges(nP);
    for (int i = 0; i < nP; ++i) {
        if (i != nP - 1) {
            add_num_edges[i] = add_size / nP;
        } else {
            add_num_edges[i] = add_size / nP + add_size % nP;
        }
    }

    // For each new edge create data wire and positional wires
    std::vector<std::vector<wire_t>> new_edge_list(nP);
    std::vector<std::vector<wire_t>> new_edge_pos_V(nP);
    std::vector<std::vector<wire_t>> new_edge_pos_S(nP);
    std::vector<std::vector<wire_t>> new_edge_pos_D(nP);

    for (int i = 0; i < nP; ++i) {
        std::vector<wire_t> new_edge_list_party(add_num_edges[i]);
        for (size_t j = 0; j < add_num_edges[i]; ++j) {
            new_edge_list_party[j] = circ.newInputWire();
        }
        new_edge_list[i] = std::move(new_edge_list_party);
    }

    // Create V^Out_i and V^In_i inputs per party (both length |V|)
    std::vector<std::vector<wire_t>> Vout(nP);
    std::vector<std::vector<wire_t>> Vin(nP);
    for (int i = 0; i < nP; ++i) {
        std::vector<wire_t> Vout_party(num_vert);
        std::vector<wire_t> Vin_party(num_vert);
        for (size_t j = 0; j < num_vert; ++j) {
            Vout_party[j] = circ.newInputWire();
            Vin_party[j] = circ.newInputWire();
        }
        Vout[i] = Vout_party;
        Vin[i] = Vin_party;
    }

    // zero wire - input 0
    auto zero_wire = circ.newInputWire();

    // Compute aggregated V_in, V_out
    std::vector<wire_t> Vout_agg(num_vert);
    std::vector<wire_t> Vin_agg(num_vert);
    for (size_t j = 0; j < num_vert; ++j) {
        // start with party 0's value then add others
        wire_t acc_out = Vout[0][j];
        for (int p = 1; p < nP; ++p) {
            acc_out = circ.addGate(common::utils::GateType::kAdd, acc_out, Vout[p][j]);
        }
        Vout_agg[j] = acc_out;

        wire_t acc_in = Vin[0][j];
        for (int p = 1; p < nP; ++p) {
            acc_in = circ.addGate(common::utils::GateType::kAdd, acc_in, Vin[p][j]);
        }
        Vin_agg[j] = acc_in;
    }

    // Compute cumulative offsets OffIn[j] and OffOut[j] = sum_{k=0}^{j-1} Vout_agg[k]
    std::vector<wire_t> OffIn(num_vert);
    std::vector<wire_t> OffOut(num_vert);
    OffIn[0] = zero_wire;
    OffOut[0] = zero_wire;
    for (size_t j = 1; j < num_vert; ++j) {
        OffIn[j] = circ.addGate(common::utils::GateType::kAdd, OffIn[j - 1], Vin_agg[j - 1]);
        OffOut[j] = circ.addGate(common::utils::GateType::kAdd, OffOut[j - 1], Vout_agg[j - 1]);
    }
    
    // For each party i, compute indicator arrays
    std::vector<std::vector<wire_t>> IIn(nP);
    std::vector<std::vector<wire_t>> IOut(nP);
    for (int i = 0; i < nP; ++i) {
        IIn[i].resize(num_vert);
        IOut[i].resize(num_vert);
        for (size_t j = 0; j < num_vert; ++j) {
            // IIn: 1 - Eqz(Vin[i][j])  --> neg_eq0 = -Eqz(Vin); IIn = neg_eq0 + 1
            auto eq0 = circ.addGate(common::utils::GateType::kEqz, Vin[i][j]);
            auto neg_eq0 = circ.addConstOpGate(common::utils::GateType::kConstMul, eq0, Ring(-1));
            IIn[i][j] = circ.addConstOpGate(common::utils::GateType::kConstAdd, neg_eq0, Ring(1));

            // IOut: 1 - Eqz(Vout[i][j] - 1)
            auto v_minus_1 = circ.addConstOpGate(common::utils::GateType::kConstAdd, Vout[i][j], Ring(-1));
            auto eq1 = circ.addGate(common::utils::GateType::kEqz, v_minus_1);
            auto neg_eq1 = circ.addConstOpGate(common::utils::GateType::kConstMul, eq1, Ring(-1));
            IOut[i][j] = circ.addConstOpGate(common::utils::GateType::kConstAdd, neg_eq1, Ring(1));
        }
    }
    
    // Precompute absolute vertex indices for each party's existing vertices
    std::vector<size_t> party_vertex_offset(nP);
    party_vertex_offset[0] = 0;
    for (int i = 1; i < nP; ++i) {
        party_vertex_offset[i] = party_vertex_offset[i-1] + subg_num_vert[i-1];
    }

    // Compute pos_S for new edges
    // For each vertex, compute data_e = σS + OffOut[j] 
    std::vector<wire_t> data_e(num_vert);
    for (int i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_vert[i]; ++j) {
            size_t abs_vertex_idx = party_vertex_offset[i] + j;
            data_e[abs_vertex_idx] = circ.addGate(common::utils::GateType::kAdd, 
                subg_vertex_pos_S[i][j], OffOut[abs_vertex_idx]);   
        }
    }
    std::vector<wire_t> data_e(num_vert);
    
    // Group-wise Propagate
    // Prepare permutations (two separate permutations: one for T1, one for T2)
    // Permutations set as identity for benchmarking
    std::vector<std::vector<int>> permutation;
    
    // T1 permutation
    std::vector<int> t1_perm(num_vert);
    for (size_t i = 0; i < num_vert; ++i) {
        t1_perm[i] = i;
    }
    
    // T2 permutation
    std::vector<int> t2_perm(add_size);
    for (size_t i = 0; i < add_size; ++i) {
        t2_perm[i] = i;
    }
    
    // For party 0, we need all parties' permutations
    // For other parties, just their own permutation
    if (pid == 0) {
        // Party 0 stores all parties' permutations
        // For simplicity, using same permutation for all parties
        permutation.push_back(t1_perm);  // T1 permutation
        permutation.push_back(t2_perm);  // T2 permutation
    } else {
        // Other parties store their own permutations
        permutation.push_back(t1_perm);  // T1 permutation
        permutation.push_back(t2_perm);  // T2 permutation
    }
    // Group-wise propagate gates
    for (int i = 0; i < nP; ++i) {

    }

    
    
    return circ;
}