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

    // Initialize vectors of vertices to be added.
    // Distribute add_size across parties 
    std::vector<size_t> add_num_vert(nP);
    for (int i = 0; i < nP; ++i) {
        if (i != nP - 1) {
            add_num_vert[i] = add_size / nP;
        } else {
            add_num_vert[i] = add_size / nP + add_size % nP;
        }
    }

    // For each new vertex we create: data wire and three positional wires (V, S, D)
    std::vector<std::vector<wire_t>> new_vertex_list(nP);
    std::vector<std::vector<wire_t>> new_vertex_pos_V(nP);
    std::vector<std::vector<wire_t>> new_vertex_pos_S(nP);
    std::vector<std::vector<wire_t>> new_vertex_pos_D(nP);

    for (int i = 0; i < nP; ++i) {
        std::vector<wire_t> new_vertex_list_party(add_num_vert[i]);
        for (int j = 0; j < add_num_vert[i]; ++j) {
            new_vertex_list_party[j] = circ.newInputWire();
        }
        new_vertex_list[i] = new_vertex_list_party;
    }

    // zero wire for assigning constant value to wire (is this ok?)
    auto zero_wire = circ.addConstOpGate(common::utils::GateType::kConstMul, subg_vertex_list[0][0], 0);

    // Compute per-party delta values (prefix sums of new vertices)
    // and prefix sums of existing vertices
    std::vector<size_t> delta_s(nP);
    std::vector<size_t> exiting_vertex_sums(nP);
    for (int i = 1; i < nP; ++i) {
        delta_s[i] = delta_s[i - 1] + add_num_vert[i - 1];
        existing_vertex_sums[i] = existing_vertex_sums[i-1] + subg_num_vert[i-1];
    }

    // Compute new vertex pos_V values
    std::vector<std::vector<size_t>> new_vertex_pos_V_values(nP);
    for (int i = 1; i < nP; ++i) {
        for (int k = 0; k < add_num_vert[i]; ++k){
            new_vertex_pos_V_values[i][k] = existing_vertex_sums[i] + delta_s[i] + k;
        }
    }
    
    // Assign positions to new vertices
    for (int i = 0; i < nP; ++i){
        std::vector<wire_t> new_vertex_pos_V_party(add_num_vert[i]);
        std::vector<wire_t> new_vertex_pos_S_party(add_num_vert[i]);
        std::vector<wire_t> new_vertex_pos_D_party(add_num_vert[i]);
        for (int k = 0; k < add_num_vert[i]; ++k){
            new_vertex_pos_V_party[k] = circ.addConstOpGate(common::utils::GateType::kConstAdd, zero_wire, new_vertex_pos_V_values[i][k]);
            new_vertex_pos_S_party[k] = 
                circ.addConstOpGate(common::utils::GateType::kConstAdd, subg_vertex_pos_S[i][subg_num_vert[i]-1], delta_s[i] + k);
            new_vertex_pos_D_party[k] = 
                circ.addConstOpGate(common::utils::GateType::kConstAdd, subg_vertex_pos_D[i][subg_num_vert[i]-1], delta_s[i] + k);
        }       
    }

    // Update positions of existing vertices and edges
    


    return circ;
}