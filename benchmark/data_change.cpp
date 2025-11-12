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

common::utils::Circuit<Ring> generateCircuit(int nP, int pid, std::vector<size_t> subg_num_vert,
                                            std::vector<size_t> subg_num_edge) {

    std::cout << "Generating circuit" << std::endl;
    
    common::utils::Circuit<Ring> circ;

    // Input wires for each client's subgraph vertices and edges
    std::vector<std::vector<wire_t>> subg_vertex_list(nP);
    for (int i = 0; i < subg_vertex_list.size(); ++i) {
        std::vector<wire_t> subg_vertex_list_party(subg_num_vert[i]);
        for (int j = 0; j < subg_vertex_list_party.size(); ++j) {
            subg_vertex_list_party[j] = circ.newInputWire();
        }
        subg_vertex_list[i] = subg_vertex_list_party;
    }
    std::vector<std::vector<wire_t>> subg_edge_list(nP);
    for (int i = 0; i < subg_edge_list.size(); ++i) {
        std::vector<wire_t> subg_edge_list_party(subg_num_edge[i]);
        // mistake in emGraph implementation?
        for (int j = 0; j < subg_edge_list_party.size(); ++j) {
            subg_edge_list_party[j] = circ.newInputWire();
        }
        subg_edge_list[i] = subg_edge_list_party;
    }

    // Initialize data change vectors. 

    std::vector<std::vector<wire_t>> vertex_changed(nP);
    for (int i = 0; i < vertex_changed.size(); ++i){
        std::vector<wire_t> vertex_changed_party(subg_num_vert[i]);
        for (int j = 0; j < vertex_changed_party.size(); ++j){
            vertex_changed_party[j] = circ.newInputWire();
        }
        vertex_changed[i] = vertex_changed_party;
    }
    std::vector<std::vector<wire_t>> edge_changed(nP);
    for (int i = 0; i < edge_changed.size(); ++i){
        std::vector<wire_t> edge_changed_party(subg_num_edge[i]);
        for (int j = 0; j < edge_changed_party.size(); ++j){
            edge_changed_party[j] = circ.newInputWire();
        }
        edge_changed[i] = edge_changed_party;
    }
    std::vector<std::vector<wire_t>> vertex_changed_data(nP);
    for (int i = 0; i < vertex_changed_data.size(); ++i){
        std::vector<wire_t> vertex_changed_data_party(subg_num_vert[i]);
        for (int j = 0; j < vertex_changed_data_party.size(); ++j){
            vertex_changed_data_party[j] = circ.newInputWire();
        }
        vertex_changed_data[i] = vertex_changed_data_party;
    }
    std::vector<std::vector<wire_t>> edge_changed_data(nP);
    for (int i = 0; i < edge_changed_data.size(); ++i){
        std::vector<wire_t> edge_changed_data_party(subg_num_edge[i]);
        for (int j = 0; j < edge_changed_data_party.size(); ++j){
            edge_changed_data_party[j] = circ.newInputWire();
        }
        edge_changed_data[i] = edge_changed_data_party;
    }

    // Update vertex data
    std::vector<std::vector<wire_t>> updated_vertex_list(nP);
    for (int i = 0; i < updated_vertex_list.size(); ++i){
        updated_vertex_list[i].resize(subg_num_vert[i]);
    }

    for (int i = 0; i < nP; ++i){
        for (int j = 0; j < vertex_changed[i].size(); ++j){
            auto diff = 
                circ.addGate(common::utils::GateType::kSub, vertex_changed_data[i][j], subg_vertex_list[i][j]);
            auto change = 
                circ.addGate(common::utils::GateType::kMul, vertex_changed[i][j], diff);
            updated_vertex_list[i][j] =
                circ.addGate(common::utils::GateType::kAdd, subg_vertex_list[i][j], change);
            circ.setAsOutput(updated_vertex_list[i][j]);
        }
    }

    // Update edge data
    std::vector<std::vector<wire_t>> updated_edge_list(nP);
    for (int i = 0; i < updated_edge_list.size(); ++i){
        updated_edge_list[i].resize(subg_num_edge[i]);
    }

    for (int i = 0; i < nP; ++i){
        for (int j = 0; j < edge_changed[i].size(); ++j){
            auto diff = 
                circ.addGate(common::utils::GateType::kSub, edge_changed_data[i][j], subg_edge_list[i][j]);
            auto change = 
                circ.addGate(common::utils::GateType::kMul, edge_changed[i][j], diff);
            updated_edge_list[i][j] =
                circ.addGate(common::utils::GateType::kAdd, subg_edge_list[i][j], change);
            circ.setAsOutput(updated_edge_list[i][j]);
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


    StatsPoint start(*network);
    network->sync();

    auto circ = generateCircuit(nP, pid, dist_daglist.VSizes, dist_daglist.ESizes).orderGatesByLevel();
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
    
    // Declare updated_daglist outside to use after output generation
    DistributedDaglist updated_daglist;
    
    if (pid == 1) {

        // Print distribution info
        std::cout << "\n=== Daglist Distribution ===" << std::endl;
        for (int i = 0; i < nC; ++i) {
            std::cout << "Client " << i << ": " << dist_daglist.VSizes[i] << " vertices, "
                    << dist_daglist.ESizes[i] << " edges" << std::endl;
        }
        std::cout << "============================\n" << std::endl;
    
        
        // Generate random data updates (change 20% of entries)
        Ring num_changes = static_cast<Ring>(vec_size * 0.2);
        updated_daglist = generate_random_data_updates(dist_daglist, num_changes, seed);

        // Set circuit inputs in the expected order:
        // 1. Original graph data (vertices + edges) across all clients
        // 2. Change indicators (isChangeV + isChangeE) across all clients
        // 3. New data values (ChangeV + ChangeE) across all clients

        // Collect all values in client order (client 0..nC-1)
        for (int c = 0; c < nC; ++c) {
            // Vertex data for this client
            for (size_t i = 0; i < updated_daglist.VSizes[c]; ++i) {
                graph_input_values.push_back(updated_daglist.VertexLists[c][i].data);
            }
            // Edge data for this client
            for (size_t i = 0; i < updated_daglist.ESizes[c]; ++i) {
                graph_input_values.push_back(updated_daglist.EdgeLists[c][i].data);
            }
        }

        for (int c = 0; c < nC; ++c) {
            // Vertex change indicators
            for (size_t i = 0; i < updated_daglist.VSizes[c]; ++i) {
                indicator_input_values.push_back(updated_daglist.isChangeV[c][i]);
            }
            // Edge change indicators
            for (size_t i = 0; i < updated_daglist.ESizes[c]; ++i) {
                indicator_input_values.push_back(updated_daglist.isChangeE[c][i]);
            }
        }

        for (int c = 0; c < nC; ++c) {
            // Vertex new data values
            for (size_t i = 0; i < updated_daglist.VSizes[c]; ++i) {
                update_input_values.push_back(updated_daglist.ChangeV[c][i]);
            }
            // Edge new data values
            for (size_t i = 0; i < updated_daglist.ESizes[c]; ++i) {
                update_input_values.push_back(updated_daglist.ChangeE[c][i]);
            }
        }

        // Map collected values into circuit input wires (in order)
        size_t wire_idx = 0;
        for (size_t i = 0; i < graph_input_values.size() && wire_idx < input_wires.size(); ++i) {
            inputs[input_wires[wire_idx++]] = graph_input_values[i];
        }
        for (size_t i = 0; i < indicator_input_values.size() && wire_idx < input_wires.size(); ++i) {
            inputs[input_wires[wire_idx++]] = indicator_input_values[i];
        }
        for (size_t i = 0; i < update_input_values.size() && wire_idx < input_wires.size(); ++i) {
            inputs[input_wires[wire_idx++]] = update_input_values[i];
        }
        

    }

    std::cout << "Total inputs set: " << inputs.size() << std::endl;
    
    eval.setInputs(inputs);
    
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

    // Update daglist with outputs and print
    if (pid == 1) {
        // Update the distributed daglist with output values
        size_t output_idx = 0;
        for (int c = 0; c < nC; ++c) {
            // Update vertex data
            for (size_t i = 0; i < updated_daglist.VSizes[c]; ++i) {
                if (output_idx < outputs.size()) {
                    updated_daglist.VertexLists[c][i].data = outputs[output_idx++];
                }
            }
            // Update edge data
            for (size_t i = 0; i < updated_daglist.ESizes[c]; ++i) {
                if (output_idx < outputs.size()) {
                    updated_daglist.EdgeLists[c][i].data = outputs[output_idx++];
                }
            }
        }
        
        // Print updated daglist entries for each client
        std::cout << "\n=== Updated Daglist Entries (First 5 per Client) ===" << std::endl;
        for (int c = 0; c < nC; ++c) {
            std::cout << "\n--- Client " << c << " ---" << std::endl;
            
            // Print first 5 vertices
            size_t v_print = std::min(static_cast<size_t>(5), static_cast<size_t>(updated_daglist.VSizes[c]));
            std::cout << "Vertices:" << std::endl;
            for (size_t i = 0; i < v_print; ++i) {
                const auto& entry = updated_daglist.VertexLists[c][i];
                std::cout << "  V[" << i << "]: "
                          << "src=" << entry.src << ", "
                          << "dst=" << entry.dst << ", "
                          << "isV=" << entry.isV << ", "
                          << "data=" << entry.data << ", "
                          << "sigs=" << entry.sigs << ", "
                          << "sigv=" << entry.sigv << ", "
                          << "sigd=" << entry.sigd << std::endl;
            }
            
            // Print first 5 edges
            size_t e_print = std::min(static_cast<size_t>(5), static_cast<size_t>(updated_daglist.ESizes[c]));
            std::cout << "Edges:" << std::endl;
            for (size_t i = 0; i < e_print; ++i) {
                const auto& entry = updated_daglist.EdgeLists[c][i];
                std::cout << "  E[" << i << "]: "
                          << "src=" << entry.src << ", "
                          << "dst=" << entry.dst << ", "
                          << "isV=" << entry.isV << ", "
                          << "data=" << entry.data << ", "
                          << "sigs=" << entry.sigs << ", "
                          << "sigv=" << entry.sigv << ", "
                          << "sigd=" << entry.sigd << std::endl;
            }
        }
        std::cout << "=============================================\n" << std::endl;
    }

    // Print example inputs and outputs
    if (pid == 1) {  // Only print from party 1 to avoid duplicate output
        std::cout << "\n=== EXAMPLE INPUTS AND OUTPUTS ===" << std::endl;
        std::cout << "\n--- Graph Data (first 10 vertices/edges) ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), graph_input_values.size()); ++i) {
            std::cout << "Element[" << i << "]: " << graph_input_values[i] << std::endl;
        }
        
        std::cout << "\n--- Update Indicators (first 10) ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), indicator_input_values.size()); ++i) {
            std::cout << "Indicator[" << i << "]: " << indicator_input_values[i] 
                      << (indicator_input_values[i] == 1 ? " (UPDATE)" : " (NO CHANGE)") << std::endl;
        }
        
        std::cout << "\n--- New Data Values (first 10) ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), update_input_values.size()); ++i) {
            std::cout << "NewData[" << i << "]: " << update_input_values[i] << std::endl;
        }
        
        std::cout << "\n--- Updated Outputs (first 10) ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), outputs.size()); ++i) {
            std::cout << "Output[" << i << "]: " << outputs[i] << std::endl;
        }
        
        std::cout << "\n--- Verification (first 5 elements) ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), std::min(graph_input_values.size(), outputs.size())); ++i) {
            Ring expected = graph_input_values[i] + 
                           indicator_input_values[i] * (update_input_values[i] - graph_input_values[i]);
            std::cout << "Element[" << i << "]: Original=" << graph_input_values[i] 
                      << ", Indicator=" << indicator_input_values[i]
                      << ", NewData=" << update_input_values[i]
                      << ", Expected=" << expected
                      << ", Actual=" << outputs[i]
                      << (expected == outputs[i] ? " ✓" : " ✗") << std::endl;
        }
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