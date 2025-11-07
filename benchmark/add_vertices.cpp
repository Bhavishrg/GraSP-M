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

    // zero wire for assigning constant value to wire
    auto zero_wire = circ.newInputWire();

    // Compute per-party delta values (prefix sums of new vertices)
    // and prefix sums of existing vertices
    std::vector<size_t> delta(nP);
    std::vector<size_t> existing_vertex_sums(nP);
    delta[0] = 0;
    existing_vertex_sums[0] = 0;
    for (int i = 1; i < nP; ++i) {
        delta[i] = delta[i - 1] + add_num_vert[i - 1];
        existing_vertex_sums[i] = existing_vertex_sums[i-1] + subg_num_vert[i-1];
    }

    // Compute new vertex pos_V values
    std::vector<std::vector<size_t>> new_vertex_pos_V_values(nP);
    for (int i = 0; i < nP; ++i) {
        new_vertex_pos_V_values[i].resize(add_num_vert[i]);
        for (int k = 0; k < add_num_vert[i]; ++k){
            new_vertex_pos_V_values[i][k] = 
                existing_vertex_sums[i] + delta[i] + subg_num_vert[i] + k;
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
                circ.addConstOpGate(common::utils::GateType::kConstAdd, subg_vertex_pos_S[i][subg_num_vert[i]-1], delta[i] + k);
            new_vertex_pos_D_party[k] = 
                circ.addConstOpGate(common::utils::GateType::kConstAdd, subg_vertex_pos_D[i][subg_num_vert[i]-1], delta[i] + k);
        }   
        new_vertex_pos_V[i] = std::move(new_vertex_pos_V_party);
        new_vertex_pos_S[i] = std::move(new_vertex_pos_S_party);
        new_vertex_pos_D[i] = std::move(new_vertex_pos_D_party);    
    }

    // Update positions of existing vertices and edges
    // Assign delta values to wires
    std::vector<wire_t> delta_s(vec_size);
    std::vector<wire_t> delta_d(vec_size);
    int index = 0;
    for (size_t i = 0 ; i < nP; ++i) {
        for (size_t j = 0; j < add_num_vert[i]; ++j) {
            delta_s[index] = circ.addConstOpGate(common::utils::GateType::kConstAdd, zero_wire, delta[i]);
            delta_d[index] = circ.addConstOpGate(common::utils::GateType::kConstAdd, zero_wire, delta[i]);
            index++;
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_edge[i]; ++j) {
            // Initialize wires to zero (don't set input)
            delta_s[index] = circ.newInputWire();
            delta_d[index] = circ.newInputWire();
            index++;
        }
    }

    // Flatten position maps
    std::vector<wire_t> pos_S(vec_size);
    std::vector<wire_t> pos_D(vec_size);
    index = 0;
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_vert[i]; ++j) {
            pos_S[index] = subg_vertex_pos_S[i][j];
            pos_D[index] = subg_vertex_pos_D[i][j];
            index++;
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_edge[i]; ++j) {
            pos_S[index] = subg_edge_pos_S[i][j];
            pos_D[index] = subg_edge_pos_D[i][j];
            index++;
        }
    }

    // Generate permutation for shuffle
    // Here we just pass identity permutations
    std::vector<int> base_perm(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        base_perm[i] = static_cast<int>(i);
    }
    std::vector<std::vector<int>> permutation;
    permutation.push_back(base_perm);
    if (pid == 0) {
        for (int i = 1; i < nP; ++i) {
            permutation.push_back(base_perm);
        }
    }

    // Propagate
    auto prop_delta_s = addSubCircPropagate(circ, pos_S, delta_s, num_vert, permutation);
    auto prop_delta_d = addSubCircPropagate(circ, pos_D, delta_d, num_vert, permutation);

    // Update position components
    std::vector<wire_t> updated_pos_S(vec_size);
    std::vector<wire_t> updated_pos_D(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        updated_pos_S[i] = circ.addGate(common::utils::GateType::kAdd, pos_S[i], prop_delta_s[i]);
        updated_pos_D[i] = circ.addGate(common::utils::GateType::kAdd, pos_D[i], prop_delta_d[i]);
    }

    // Set outputs
    index = 0;
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_vert[i]; ++j) {
            circ.setAsOutput(subg_vertex_list[i][j]);
            circ.setAsOutput(subg_vertex_pos_V[i][j]);
            circ.setAsOutput(updated_pos_S[index]);
            circ.setAsOutput(updated_pos_D[index]);
            index++;
        }
        for (size_t j = 0; j < add_num_vert[i]; ++j) {
            circ.setAsOutput(new_vertex_list[i][j]);
            circ.setAsOutput(new_vertex_pos_V[i][j]);
            circ.setAsOutput(new_vertex_pos_S[i][j]);
            circ.setAsOutput(new_vertex_pos_D[i][j]);
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_edge[i]; ++j) {
            circ.setAsOutput(subg_edge_list[i][j]);
            circ.setAsOutput(subg_edge_pos_V[i][j]);
            circ.setAsOutput(updated_pos_S[index]);
            circ.setAsOutput(updated_pos_D[index]);
            index++;
        }
    }

    return circ;
}