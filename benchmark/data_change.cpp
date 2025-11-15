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
#include "graphutils.h"

using namespace graphdb;
using json = nlohmann::json;
namespace bpo = boost::program_options;


common::utils::Circuit<Ring> generateCircuit(int nP, int pid, DistributedDaglist dist_daglist) {

    int nC = dist_daglist.num_clients;
    auto VSizes = dist_daglist.VSizes;
    auto ESizes = dist_daglist.ESizes;

    std::cout << "Generating circuit" << std::endl;

    common::utils::Circuit<Ring> circ; 

    // Initialize all daglist field values
    std::vector<std::vector<wire_t>> vertex_src_values(nC);
    std::vector<std::vector<wire_t>> vertex_dst_values(nC);
    std::vector<std::vector<wire_t>> vertex_isV_values(nC);
    std::vector<std::vector<wire_t>> vertex_data_values(nC);
    std::vector<std::vector<wire_t>> vertex_sigs_values(nC);
    std::vector<std::vector<wire_t>> vertex_sigv_values(nC);
    std::vector<std::vector<wire_t>> vertex_sigd_values(nC);
    std::vector<std::vector<wire_t>> vertex_changed(nC);
    std::vector<std::vector<wire_t>> vertex_changed_data(nC);

    std::vector<std::vector<wire_t>> edge_src_values(nC);
    std::vector<std::vector<wire_t>> edge_dst_values(nC);
    std::vector<std::vector<wire_t>> edge_isV_values(nC);
    std::vector<std::vector<wire_t>> edge_data_values(nC);
    std::vector<std::vector<wire_t>> edge_sigs_values(nC);
    std::vector<std::vector<wire_t>> edge_sigv_values(nC);
    std::vector<std::vector<wire_t>> edge_sigd_values(nC);
    std::vector<std::vector<wire_t>> edge_changed(nC);
    std::vector<std::vector<wire_t>> edge_changed_data(nC);

    for (int i = 0; i < nC; ++i) {
        std::vector<wire_t> subg_vertex_src_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_dst_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_isV_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_data_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_sigs_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_sigv_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_sigd_values(VSizes[i]);
        std::vector<wire_t> subg_vertex_changed(VSizes[i]);
        std::vector<wire_t> subg_vertex_changed_data(VSizes[i]);
        
        std::vector<wire_t> subg_edge_src_values(ESizes[i]);
        std::vector<wire_t> subg_edge_dst_values(ESizes[i]);
        std::vector<wire_t> subg_edge_isV_values(ESizes[i]);
        std::vector<wire_t> subg_edge_data_values(ESizes[i]);
        std::vector<wire_t> subg_edge_sigs_values(ESizes[i]);
        std::vector<wire_t> subg_edge_sigv_values(ESizes[i]);
        std::vector<wire_t> subg_edge_sigd_values(ESizes[i]);
        std::vector<wire_t> subg_edge_changed(ESizes[i]);
        std::vector<wire_t> subg_edge_changed_data(ESizes[i]);

        for (int j = 0; j < VSizes[i]; ++j){
            subg_vertex_src_values[j] = circ.newInputWire();
            subg_vertex_dst_values[j] = circ.newInputWire();
            subg_vertex_isV_values[j] = circ.newInputWire();
            subg_vertex_data_values[j] = circ.newInputWire();
            subg_vertex_sigs_values[j] = circ.newInputWire();
            subg_vertex_sigv_values[j] = circ.newInputWire();
            subg_vertex_sigd_values[j] = circ.newInputWire();
            subg_vertex_changed[j] = circ.newInputWire();
            subg_vertex_changed_data[j] = circ.newInputWire();
        }

        for (int j = 0; j < ESizes[i]; ++j){
            subg_edge_src_values[j] = circ.newInputWire();
            subg_edge_dst_values[j] = circ.newInputWire();
            subg_edge_isV_values[j] = circ.newInputWire();
            subg_edge_data_values[j] = circ.newInputWire();
            subg_edge_sigs_values[j] = circ.newInputWire();
            subg_edge_sigv_values[j] = circ.newInputWire();
            subg_edge_sigd_values[j] = circ.newInputWire();
            subg_edge_changed[j] = circ.newInputWire();
            subg_edge_changed_data[j] = circ.newInputWire();
        }

        vertex_src_values[i] = subg_vertex_src_values;
        vertex_dst_values[i] = subg_vertex_dst_values;
        vertex_isV_values[i] = subg_vertex_isV_values;
        vertex_data_values[i] = subg_vertex_data_values;
        vertex_sigs_values[i] = subg_vertex_sigs_values;
        vertex_sigv_values[i] = subg_vertex_sigv_values;
        vertex_sigd_values[i] = subg_vertex_sigd_values;
        vertex_changed[i] = subg_vertex_changed;
        vertex_changed_data[i] = subg_vertex_changed_data;

        edge_src_values[i] = subg_edge_src_values;
        edge_dst_values[i] = subg_edge_dst_values;
        edge_isV_values[i] = subg_edge_isV_values;
        edge_data_values[i] = subg_edge_data_values;
        edge_sigs_values[i] = subg_edge_sigs_values;
        edge_sigv_values[i] = subg_edge_sigv_values;
        edge_sigd_values[i] = subg_edge_sigd_values;
        edge_changed[i] = subg_edge_changed;
        edge_changed_data[i] = subg_edge_changed_data;
    
    }

    // Update vertex data
    std::vector<std::vector<wire_t>> updated_vertex_list(nC);
    std::vector<std::vector<wire_t>> updated_edge_list(nC);
    for (int i = 0; i < nC; ++i){
        updated_vertex_list[i].resize(VSizes[i]);
        updated_edge_list[i].resize(ESizes[i]);
    }

    for (int i = 0; i < nC; ++i){
        for (int j = 0; j < VSizes[i]; ++j){
            auto diff = 
                circ.addGate(common::utils::GateType::kSub, vertex_changed_data[i][j], vertex_data_values[i][j]);
            auto change = 
                circ.addGate(common::utils::GateType::kMul, vertex_changed[i][j], diff);
            updated_vertex_list[i][j] =
                circ.addGate(common::utils::GateType::kAdd, vertex_data_values[i][j], change);
        }
    }

    for (int i = 0; i < nC; ++i){
        for (int j = 0; j < ESizes[i]; ++j){
            auto diff = 
                circ.addGate(common::utils::GateType::kSub, edge_changed_data[i][j], edge_data_values[i][j]);
            auto change = 
                circ.addGate(common::utils::GateType::kMul, edge_changed[i][j], diff);
            updated_edge_list[i][j] =
                circ.addGate(common::utils::GateType::kAdd, edge_data_values[i][j], change);
        }
    }

    for (int i = 0; i < nC; ++i){

        for (int j = 0; j < VSizes[i]; ++j){
            circ.setAsOutput(vertex_src_values[i][j]);
            circ.setAsOutput(vertex_dst_values[i][j]);
            circ.setAsOutput(vertex_isV_values[i][j]);
            circ.setAsOutput(updated_vertex_list[i][j]);
            circ.setAsOutput(vertex_sigs_values[i][j]);
            circ.setAsOutput(vertex_sigv_values[i][j]);
            circ.setAsOutput(vertex_sigd_values[i][j]);
        }

        for (int j = 0; j < ESizes[i]; ++j){
            circ.setAsOutput(edge_src_values[i][j]);
            circ.setAsOutput(edge_dst_values[i][j]);
            circ.setAsOutput(edge_isV_values[i][j]);
            circ.setAsOutput(updated_edge_list[i][j]);
            circ.setAsOutput(edge_sigs_values[i][j]);
            circ.setAsOutput(edge_sigv_values[i][j]);
            circ.setAsOutput(edge_sigd_values[i][j]);
        }
    }

    return circ;
}

void benchmark(const bpo::variables_map& opts) {

    bool save_output = false;
    std::string save_file;
    if (opts.count("output") != 0) {
        save_output = true;
        save_file = opts["output"].as<std::string>();
    }

    auto num_vert = opts["num-vert"].as<size_t>();
    auto num_edge = opts["num-edge"].as<size_t>();
    auto vec_size = num_vert + num_edge;
    auto nP = opts["num-parties"].as<int>();
    auto nC = opts["num-clients"].as<int>();
    auto latency = opts["latency"].as<double>();
    auto pid = opts["pid"].as<size_t>();
    auto threads = opts["threads"].as<size_t>();
    auto seed = opts["seed"].as<size_t>();
    auto repeat = opts["repeat"].as<size_t>();
    auto port = opts["port"].as<int>();
    auto use_pking = opts["use-pking"].as<bool>();

    omp_set_nested(1);
    if (nP < 10) { omp_set_num_threads(nP); }
    else { omp_set_num_threads(10); }

    std::cout << "Starting benchmarks" << std::endl;

    std::string net_config = opts.count("net-config") ? opts["net-config"].as<std::string>() : "";
    std::shared_ptr<io::NetIOMP> network = createNetwork(pid, nP, latency, port,
                                                          opts["localhost"].as<bool>(),
                                                          net_config);

    json output_data;
    output_data["details"] = {{"num_parties", nP},
                              {"num_clients", nC},
                              {"num_vert", num_vert},
                              {"num_vert", num_vert},
                              {"latency (ms)", latency},
                              {"pid", pid},
                              {"threads", threads},
                              {"seed", seed},
                              {"repeat", repeat}};
    output_data["benchmarks"] = json::array();

    std::cout << "--- Details ---" << std::endl;
    for (const auto& [key, value] : output_data["details"].items()) {
        std::cout << key << ": " << value << std::endl;
    }
    std::cout << std::endl;

    // Generate test graph
    Ring nV = static_cast<Ring>(num_vert);
    Ring nE = static_cast<Ring>(num_edge);
    
    std::cout << "============================\n" << std::endl;
    std::cout << "Generating random inputs " << std::endl;
    std::cout << "Generating scale-free graph with nV=" << nV << ", nE=" << nE << std::endl;
    auto edges = generate_scale_free(nV, nE);
    std::cout << "Generated " << edges.size() << " edges" << std::endl;
    
    std::cout << "Building daglist..." << std::endl;
    auto daglist = build_daglist(nV, edges);
    std::cout << "Built daglist with " << daglist.size() << " entries" << std::endl;
    
    // Distribute daglist across clients
    std::cout << "Distributing daglist across " << nC << " clients..." << std::endl;
    auto dist_daglist = distribute_daglist(daglist, nC);
    
    // Generate random data updates (change 20% of entries)
    Ring num_changes = static_cast<Ring>(vec_size * 0.2);
    std::cout << "Generating random data updates for " << num_changes << " entries..." << std::endl;
    dist_daglist = generate_random_data_updates(dist_daglist, num_changes, seed);


    StatsPoint start(*network);
    network->sync();

    auto circ = generateCircuit(nP, pid, dist_daglist).orderGatesByLevel();
    network->sync();

    std::cout << "--- Circuit ---" << std::endl;
    std::cout << circ << std::endl;
    
    std::unordered_map<common::utils::wire_t, int> input_pid_map;
    for (const auto& g : circ.gates_by_level[0]) {
        if (g->type == common::utils::GateType::kInp) {
            input_pid_map[g->out] = 1;  // All inputs belong to party 1
        }
    }

    std::cout << "Starting preprocessing" << std::endl;
    StatsPoint preproc_start(*network);
    // emp::PRG prg(&emp::zero_block, seed);
    int latency_us = static_cast<int>(latency * 1000);  // Convert ms to microseconds
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed, latency_us);
    auto preproc = off_eval.run(input_pid_map);
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    StatsPoint preproc_end(*network);

    std::cout << "Setting inputs" << std::endl;
    // Why latency_us and use_pking?
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed, latency_us, use_pking);
    
    std::unordered_map<common::utils::wire_t, Ring> inputs;
    
    // Collect all input wires owned by this party
    std::vector<common::utils::wire_t> input_wires;
    for (const auto& [wire, owner] : input_pid_map) {
        if (owner == static_cast<int>(pid)) {
            input_wires.push_back(wire);
        }
    }
    
    // Sort to ensure consistent ordering
    std::sort(input_wires.begin(), input_wires.end());
    
    std::cout << "Setting inputs for party " << pid << std::endl;
    
    // Only party 1 sets inputs
    std::vector<Ring> graph_input_values;
    std::vector<Ring> indicator_input_values;
    std::vector<Ring> update_input_values;

    if (pid == 1) {

        // Print distribution info
        std::cout << "\n=== Daglist Distribution ===" << std::endl;
        for (int i = 0; i < nC; ++i) {
            std::cout << "Client " << i << ": " << dist_daglist.VSizes[i] << " vertices, "
                    << dist_daglist.ESizes[i] << " edges" << std::endl;
        }
        std::cout << "============================\n" << std::endl;
    
        std::vector<Ring> all_input_values;
        
    // Collect all vertex and edge fields for all clients
    for (int c = 0; c < nC; ++c) {
        for (size_t i = 0; i < dist_daglist.VSizes[c]; ++i) {
            all_input_values.push_back(dist_daglist.VertexLists[c][i].src);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].dst);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].isV);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].data);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].sigs);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].sigv);
            all_input_values.push_back(dist_daglist.VertexLists[c][i].sigd);
            all_input_values.push_back(dist_daglist.isChangeV[c][i]);
            all_input_values.push_back(dist_daglist.ChangeV[c][i]);
        }

        for (size_t i = 0; i < dist_daglist.ESizes[c]; ++i) {
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].src);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].dst);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].isV);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].data);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].sigs);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].sigv);
            all_input_values.push_back(dist_daglist.EdgeLists[c][i].sigd);
            all_input_values.push_back(dist_daglist.isChangeE[c][i]);
            all_input_values.push_back(dist_daglist.ChangeE[c][i]);
        }
    }
        
        // Map collected values into circuit input wires (in order)
        size_t wire_idx = 0;
        for (size_t i = 0; i < all_input_values.size() && wire_idx < input_wires.size(); ++i) {
            inputs[input_wires[wire_idx++]] = all_input_values[i];
        }
        
        // Store for verification
        graph_input_values = all_input_values;
        
        std::cout << "\n=== DEBUG: First 20 inputs being set ===" << std::endl;
        size_t debug_count = std::min(size_t(20), all_input_values.size());
        for (size_t i = 0; i < debug_count; ++i) {
            std::cout << "Input[" << i << "] = " << all_input_values[i] << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }

    std::cout << "Total inputs set by party " << pid << ": " << inputs.size() << std::endl;
    
    if (pid == 1) {
        std::cout << "Party 1 setting " << inputs.size() << " actual input values" << std::endl;
    } else {
        std::cout << "Party " << pid << " setting " << inputs.size() << " empty inputs (participant in MPC)" << std::endl;
    }
    
    eval.setInputs(inputs);
    network->sync();
    
    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }
    network->flush();
    network->sync();
    StatsPoint online_end(*network);
    std::cout << "Online evaluation complete" << std::endl;

    std::cout << "Getting outputs..." << std::endl;
    network->flush();
    auto outputs = eval.getOutputs();
    network->sync();
    std::cout << "Number of outputs: " << outputs.size() << std::endl;
    


    if (pid == 1 && outputs.size() > 0) {
        std::cout << "\n=== DEBUG: First 20 raw outputs ===" << std::endl;
        for (size_t i = 0; i < std::min(size_t(20), outputs.size()); ++i) {
            std::cout << "Output[" << i << "] = " << outputs[i] << std::endl;
        }
        std::cout << "===================================\n" << std::endl;
    }


    // Update the distributed daglist with output values
    // Outputs are interleaved per entry: for each vertex/edge, all 7 fields (src, dst, isV, data, sigs, sigv, sigd) appear consecutively
    size_t output_idx = 0;
    
    for (int c = 0; c < nC; ++c) {
        // Parse vertex outputs - fields are interleaved per vertex
        for (size_t i = 0; i < dist_daglist.VSizes[c]; ++i) {
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].src = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].dst = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].isV = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].data = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].sigs = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].sigv = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.VertexLists[c][i].sigd = outputs[output_idx++];
        }
        
        // Parse edge outputs - fields are interleaved per edge
        for (size_t i = 0; i < dist_daglist.ESizes[c]; ++i) {
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].src = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].dst = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].isV = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].data = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].sigs = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].sigv = outputs[output_idx++];
            if (output_idx < outputs.size()) dist_daglist.EdgeLists[c][i].sigd = outputs[output_idx++];
        }
    }

    // Print example inputs and outputs
    if (pid == 1) {  // Only print from party 1 to avoid duplicate output
        std::cout << "\n=== EXAMPLE INPUTS AND OUTPUTS ===" << std::endl;
        
        // Input structure: interleaved per vertex/edge (9 fields each: src, dst, isV, data, sigs, sigv, sigd, isChange, newData)
        // Output structure: interleaved per vertex/edge (7 fields each: src, dst, isV, data, sigs, sigv, sigd)
        
        std::cout << "\n--- Vertex Data Updates (First 5 from Client 0) ---" << std::endl;
        
        if (nC > 0 && dist_daglist.VSizes[0] > 0) {
            size_t print_count = std::min(size_t(5), static_cast<size_t>(dist_daglist.VSizes[0]));
            
            for (size_t i = 0; i < print_count; ++i) {
                // Input: 9 fields per vertex (interleaved)
                size_t input_base = i * 9;
                size_t output_base = i * 7;
                
                if (input_base + 8 < graph_input_values.size() && output_base + 6 < outputs.size()) {
                    Ring in_src = graph_input_values[input_base + 0];
                    Ring in_dst = graph_input_values[input_base + 1];
                    Ring in_isV = graph_input_values[input_base + 2];
                    Ring in_data = graph_input_values[input_base + 3];
                    Ring in_sigs = graph_input_values[input_base + 4];
                    Ring in_sigv = graph_input_values[input_base + 5];
                    Ring in_sigd = graph_input_values[input_base + 6];
                    Ring indicator = graph_input_values[input_base + 7];
                    Ring new_data = graph_input_values[input_base + 8];
                    
                    Ring out_src = outputs[output_base + 0];
                    Ring out_dst = outputs[output_base + 1];
                    Ring out_isV = outputs[output_base + 2];
                    Ring out_data = outputs[output_base + 3];
                    Ring out_sigs = outputs[output_base + 4];
                    Ring out_sigv = outputs[output_base + 5];
                    Ring out_sigd = outputs[output_base + 6];
                    
                    Ring expected_data = in_data + indicator * (new_data - in_data);
                    
                    std::cout << "  Vertex[" << i << "]: "
                              << "src=" << in_src << "→" << out_src
                              << ", dst=" << in_dst << "→" << out_dst
                              << ", isV=" << in_isV << "→" << out_isV
                              << ", data=" << in_data << "→" << out_data
                              << " (expected=" << expected_data << (expected_data == out_data ? " ✓" : " ✗") << ")"
                              << ", Changed=" << (indicator == 1 ? "YES" : "NO")
                              << std::endl;
                }
            }
        }
        
        std::cout << "\n--- Edge Data Updates (First 5 from Client 0) ---" << std::endl;
        
        if (nC > 0 && dist_daglist.ESizes[0] > 0) {
            // Edges start after all vertices in input array
            size_t input_edge_base = dist_daglist.VSizes[0] * 9;
            size_t output_edge_base = dist_daglist.VSizes[0] * 7;
            
            size_t print_count = std::min(size_t(5), static_cast<size_t>(dist_daglist.ESizes[0]));
            
            for (size_t i = 0; i < print_count; ++i) {
                // Input: 9 fields per edge (interleaved)
                size_t input_base = input_edge_base + i * 9;
                size_t output_base = output_edge_base + i * 7;
                
                if (input_base + 8 < graph_input_values.size() && output_base + 6 < outputs.size()) {
                    Ring in_src = graph_input_values[input_base + 0];
                    Ring in_dst = graph_input_values[input_base + 1];
                    Ring in_isV = graph_input_values[input_base + 2];
                    Ring in_data = graph_input_values[input_base + 3];
                    Ring in_sigs = graph_input_values[input_base + 4];
                    Ring in_sigv = graph_input_values[input_base + 5];
                    Ring in_sigd = graph_input_values[input_base + 6];
                    Ring indicator = graph_input_values[input_base + 7];
                    Ring new_data = graph_input_values[input_base + 8];
                    
                    Ring out_src = outputs[output_base + 0];
                    Ring out_dst = outputs[output_base + 1];
                    Ring out_isV = outputs[output_base + 2];
                    Ring out_data = outputs[output_base + 3];
                    Ring out_sigs = outputs[output_base + 4];
                    Ring out_sigv = outputs[output_base + 5];
                    Ring out_sigd = outputs[output_base + 6];
                    
                    Ring expected_data = in_data + indicator * (new_data - in_data);
                    
                    std::cout << "  Edge[" << i << "]: "
                              << "src=" << in_src << "→" << out_src
                              << ", dst=" << in_dst << "→" << out_dst
                              << ", isV=" << in_isV << "→" << out_isV
                              << ", data=" << in_data << "→" << out_data
                              << " (expected=" << expected_data << (expected_data == out_data ? " ✓" : " ✗") << ")"
                              << ", Changed=" << (indicator == 1 ? "YES" : "NO")
                              << std::endl;
                }
            }
        }
        
        size_t total_vertices = 0;
        size_t total_edges = 0;
        for (int c = 0; c < nC; ++c) {
            total_vertices += dist_daglist.VSizes[c];
            total_edges += dist_daglist.ESizes[c];
        }
        
        std::cout << "\n--- Summary ---" << std::endl;
        std::cout << "Total inputs: " << graph_input_values.size() << std::endl;
        std::cout << "Total outputs: " << outputs.size() << std::endl;
        std::cout << "Total vertices: " << total_vertices << std::endl;
        std::cout << "Total edges: " << total_edges << std::endl;
        std::cout << "Number of clients: " << nC << std::endl;
        std::cout << "===================================\n" << std::endl;
    }

    StatsPoint end(*network);

    auto preproc_rbench = preproc_end - preproc_start;
    auto online_rbench = online_end - online_start;
    auto total_rbench = end - start;
    output_data["benchmarks"].push_back(preproc_rbench);
    output_data["benchmarks"].push_back(online_rbench);
    output_data["benchmarks"].push_back(total_rbench);

    size_t pre_bytes_sent = 0;
    for (const auto& val : preproc_rbench["communication"]) {
        pre_bytes_sent += val.get<int64_t>();
    }
    size_t online_bytes_sent = 0;
    for (const auto& val : online_rbench["communication"]) {
        online_bytes_sent += val.get<int64_t>();
    }
    size_t total_bytes_sent = 0;
    for (const auto& val : total_rbench["communication"]) {
        total_bytes_sent += val.get<int64_t>();
    }

    std::cout << "preproc time: " << preproc_rbench["time"] << " ms" << std::endl;
    std::cout << "preproc sent: " << pre_bytes_sent << " bytes" << std::endl;
    std::cout << "online time: " << online_rbench["time"] << " ms" << std::endl;
    std::cout << "online sent: " << online_bytes_sent << " bytes" << std::endl;
    std::cout << "total time: " << total_rbench["time"] << " ms" << std::endl;
    std::cout << "total sent: " << total_bytes_sent << " bytes" << std::endl;
    std::cout << std::endl;

    output_data["stats"] = {{"peak_virtual_memory", peakVirtualMemory()},
                            {"peak_resident_set_size", peakResidentSetSize()}};

    std::cout << "--- Statistics ---" << std::endl;
    for (const auto& [key, value] : output_data["stats"].items()) {
        std::cout << key << ": " << value << std::endl;
    }
    std::cout << std::endl;

    if (save_output) {
        saveJson(output_data, save_file);
    }
}

// clang-format off
bpo::options_description programOptions() {
    bpo::options_description desc("Following options are supported by config file too.");
    desc.add_options()
        ("num-parties,n", bpo::value<int>()->required(), "Number of parties.")
        ("num-clients", bpo::value<int>()->default_value(2), "Number of parties.")
        ("num-vert", bpo::value<size_t>()->default_value(1000), "Number of vertices in the graph.")
        ("num-edge", bpo::value<size_t>()->default_value(4000), "Number of edges in the graph.")
        ("num-payloads", bpo::value<size_t>()->default_value(1), "Number of payload vectors.")
        ("latency,l", bpo::value<double>()->default_value(0.5), "Network latency in ms.")
        ("pid,p", bpo::value<size_t>()->required(), "Party ID.")
        ("threads,t", bpo::value<size_t>()->default_value(6), "Number of threads (recommended 6).")
        ("seed", bpo::value<size_t>()->default_value(200), "Value of the random seed.")
        ("net-config", bpo::value<std::string>(), "Path to JSON file containing network details of all parties.")
        ("localhost", bpo::bool_switch(), "All parties are on same machine.")
        ("port", bpo::value<int>()->default_value(10000), "Base port for networking.")
        ("output,o", bpo::value<std::string>(), "File to save benchmarks.")
        ("repeat,r", bpo::value<size_t>()->default_value(1), "Number of times to run benchmarks.")
        ("use-pking", bpo::value<bool>()->default_value(true), "Use king party for reconstruction (true) or direct reconstruction (false).");
  return desc;
}
// clang-format on

int main(int argc, char* argv[]) {
    auto prog_opts(programOptions());
    bpo::options_description cmdline("Benchmark secure compaction circuit.");
    cmdline.add(prog_opts);
    cmdline.add_options()(
      "config,c", bpo::value<std::string>(),
      "configuration file for easy specification of cmd line arguments")(
      "help,h", "produce help message");
    bpo::variables_map opts;
    bpo::store(bpo::command_line_parser(argc, argv).options(cmdline).run(), opts);
    if (opts.count("help") != 0) {
        std::cout << cmdline << std::endl;
        return 0;
    }
    if (opts.count("config") > 0) {
        std::string cpath(opts["config"].as<std::string>());
        std::ifstream fin(cpath.c_str());
        if (fin.fail()) {
            std::cerr << "Could not open configuration file at " << cpath << std::endl;
            return 1;
        }
        bpo::store(bpo::parse_config_file(fin, prog_opts), opts);
    }
    try {
        bpo::notify(opts);
        if (!opts["localhost"].as<bool>() && (opts.count("net-config") == 0)) {
            throw std::runtime_error("Expected one of 'localhost' or 'net-config'");
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    try {
        benchmark(opts);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\nFatal error" << std::endl;
        return 1;
    }
    return 0;
}

// usage: ./../run.sh data_change --num-parties 2 --num-clients 2 --num-vert 1000 --num-edge 4000