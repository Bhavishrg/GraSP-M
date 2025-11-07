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

common::utils::Circuit<Ring> generateCircuit(int nP, int pid, size_t vec_size, size_t del_v_size, size_t del_e_size) {

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

    std::vector<size_t> del_num_edges(nP);
    for (int i = 0; i < nP; ++i) {
        if (i != nP - 1) {
            del_num_edges[i] = del_e_size / nP;
        } else {
            del_num_edges[i] = del_e_size / nP + del_e_size % nP;
        }
    }

    std::vector<size_t> del_num_vert(nP);
    for (int i = 0; i < nP; ++i) {
        if (i != nP - 1) {
            del_num_vert[i] = del_v_size / nP;
        } else {
            del_num_vert[i] = del_v_size / nP + del_v_size % nP;
        }
    }

    // Initialize delete vectors. 
    //Assume GenList is done.

    std::vector<std::vector<wire_t>> vertex_del(nP);
    for (int i = 0; i < vertex_del.size(); ++i){
        std::vector<wire_t> vertex_del_party(subg_num_vert[i]);
        for (int j = 0; j < vertex_del_party.size(); ++j){
            vertex_del_party[j] = circ.newInputWire();
        }
        vertex_del[i] = vertex_del_party;
    }
    std::vector<std::vector<wire_t>> edge_del(nP);
    for (int i = 0; i < edge_del.size(); ++i){
        std::vector<wire_t> edge_del_party(subg_num_edge[i]);
        for (int j = 0; j < edge_del_party.size(); ++j){
            edge_del_party[j] = circ.newInputWire();
        }
        edge_del[i] = edge_del_party;
    }

    // Generate permutation for shuffle
    // Here we just pass identity permutations
    std::vector<int> base_perm(num_vert + num_edge);
    for (size_t i = 0; i < num_vert + num_edge; ++i) {
        base_perm[i] = static_cast<int>(i);
    }
    std::vector<std::vector<int>> permutation;
    permutation.push_back(base_perm);
    if (pid == 0) {
        for (int i = 1; i < nP; ++i) {
            permutation.push_back(base_perm);
        }
    }

    // Flatten del array 
    std::vector<wire_t> del(num_vert + num_edge);
    int index = 0;
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_vert[i]; ++j) {
            del[index] = vertex_del[i][j];
            index++;
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_edge[i]; ++j) {
            del[index] = edge_del[i][j];
            index++;
        }
    }

    // Flatten G
    std::vector<wire_t> graph_data(num_vert + num_edge);
    std::vector<wire_t> pos_V(num_vert + num_edge);
    std::vector<wire_t> pos_S(num_vert + num_edge);
    std::vector<wire_t> pos_D(num_vert + num_edge);
    index = 0;
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_vert[i]; ++j) {
            graph_data[index] = subg_vertex_list[i][j];
            pos_V[index] = subg_vertex_pos_V[i][j];
            pos_S[index] = subg_vertex_pos_S[i][j];
            pos_D[index] = subg_vertex_pos_D[i][j];
            index++;
        }
    }
    for (size_t i = 0; i < nP; ++i) {
        for (size_t j = 0; j < subg_num_edge[i]; ++j) {
            graph_data[index] = subg_edge_list[i][j];
            pos_V[index] = subg_edge_pos_V[i][j];
            pos_S[index] = subg_edge_pos_S[i][j];
            pos_D[index] = subg_edge_pos_D[i][j];
            index++;
        }
    }

    // Propagate del tag
    auto del_S = addSubCircPropagate(circ, pos_S, del, num_vert, permutation);
    auto del_D = addSubCircPropagate(circ, pos_D, del, num_vert, permutation);

    // Combine del tags
    std::vector<wire_t> del_final(num_vert + num_edge);
    for (size_t i = 0; i < num_vert + num_edge; ++i) {
        auto temp = circ.addGate(common::utils::GateType::kAdd, del_S[i], del_D[i]);
        temp = circ.addGate(common::utils::GateType::kAdd, del[i], temp);
        temp = circ.addGate(common::utils::GateType::kEqz, temp);
        temp = circ.addConstOpGate(common::utils::GateType::kConstMul, temp, Ring(-1));
        del_final[i] = circ.addConstOpGate(common::utils::GateType::kConstAdd, temp, Ring(1));
    }

    // Update pos_V
    std::vector<wire_t> updated_pos_V(vec_size);
    updated_pos_V[0] = pos_V[0];
    wire_t prefix_sum = del_final[0];
    for (size_t i = 1; i < vec_size; ++i) {
        updated_pos_V[i] = circ.addGate(common::utils::GateType::kSub, pos_V[i], prefix_sum);
        prefix_sum = circ.addGate(common::utils::GateType::kAdd, prefix_sum, del_final[i]);
    }

    // Combine graph vectors into single payload
    std::vector<std::vector<wire_t>> payload;
    payload.reserve(5);
    payload.push_back(graph_data);       // payload[0]
    payload.push_back(updated_pos_V);    // payload[1]
    payload.push_back(pos_S);            // payload[2]
    payload.push_back(pos_D);            // payload[3]
    payload.push_back(del_final);        // payload[4]

    // Reorder to source order
    auto payload_s = addSubCircPermList(circ, payload[2], payload, permutation);

    // Update pos_S
    std::vector<wire_t> updated_pos_S(vec_size);
    std::vector<wire_t> pos_S_s = payload_s[2];
    std::vector<wire_t> del_s = payload_s[4];
    updated_pos_S[0] = pos_S_s[0];
    prefix_sum = del_s[0];
    for (size_t i = 1; i < vec_size; ++i) {
        updated_pos_S[i] = circ.addGate(common::utils::GateType::kSub, pos_S_s[i], prefix_sum);
        prefix_sum = circ.addGate(common::utils::GateType::kAdd, prefix_sum, del_s[i]);
    }
    payload_s[2] = updated_pos_S; 

    // Reorder to destination order
    auto payload_d = addSubCircPermList(circ, payload_s[3], payload_s, permutation);

    // Update pos_D
    std::vector<wire_t> updated_pos_D(vec_size);
    std::vector<wire_t> pos_D_d = payload_d[3];
    std::vector<wire_t> del_d = payload_d[4];
    updated_pos_D[0] = pos_D_d[0];
    prefix_sum = del_d[0];
    for (size_t i = 1; i < vec_size; ++i) {
        updated_pos_D[i] = circ.addGate(common::utils::GateType::kSub, pos_D_d[i], prefix_sum);
        prefix_sum = circ.addGate(common::utils::GateType::kAdd, prefix_sum, del_d[i]);
    }
    payload_d[3] = updated_pos_D; 
    
    // Random Shuffle
    auto shuffled_data = circ.addMGate(common::utils::GateType::kShuffle, payload_d[0], permutation);
    auto shuffled_pos_V = circ.addMGate(common::utils::GateType::kShuffle, payload_d[1], permutation);
    auto shuffled_pos_S = circ.addMGate(common::utils::GateType::kShuffle, payload_d[2], permutation);
    auto shuffled_pos_D = circ.addMGate(common::utils::GateType::kShuffle, payload_d[3], permutation);
    auto shuffled_del = circ.addMGate(common::utils::GateType::kShuffle, payload_d[4], permutation);

    // Set outputs
    for (size_t i = 0; i < vec_size; ++i) {
        circ.setAsOutput(shuffled_data[i]);
        circ.setAsOutput(shuffled_pos_V[i]);
        circ.setAsOutput(shuffled_pos_S[i]);
        circ.setAsOutput(shuffled_pos_D[i]);
        circ.setAsOutput(shuffled_del[i]);
    }

    return circ;
}
