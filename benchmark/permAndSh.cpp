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

common::utils::Circuit<Ring> generateCircuit(int nP, int pid, size_t vec_size) {

    std::cout << "Generating circuit for PermAndSh" << std::endl;

    // Create identity permutation for each party
    std::vector<int> base_perm(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        base_perm[i] = static_cast<int>(i);
    }

    common::utils::Circuit<Ring> circ;

    // Create nP parallel instances of kPermAndSh, one for each party as owner
    std::vector<std::vector<common::utils::wire_t>> all_outputs;
    
    for (int owner = 1; owner <= nP; ++owner) {
        // Each instance has its own input vector
        std::vector<common::utils::wire_t> input_vector(vec_size);
        std::generate(input_vector.begin(), input_vector.end(), [&]() { return circ.newInputWire(); });

        // Create permutation for this owner
        std::vector<std::vector<int>> permutation = {base_perm};
        
        // PermAndSh returns vec_size outputs (permuted shares for the owner)
        auto outputs = circ.addMGate(common::utils::GateType::kPermAndSh,  input_vector, permutation, owner);
        
        all_outputs.push_back(outputs);
    }
    
    // Set all outputs as circuit outputs
    for (int owner = 0; owner < nP; ++owner) {
        for (size_t i = 0; i < vec_size; ++i) {
            circ.setAsOutput(all_outputs[owner][i]);
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

    auto nP = opts["num-parties"].as<int>();
    auto vec_size = opts["vec-size"].as<size_t>();
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
    std::cout << "Starting benchmarks for PermAndSh" << std::endl;

    std::string net_config = opts.count("net-config") ? opts["net-config"].as<std::string>() : "";
    std::shared_ptr<io::NetIOMP> network = createNetwork(pid, nP, latency, port,
                                                          opts["localhost"].as<bool>(),
                                                          net_config);

    // Increase socket buffer sizes to prevent deadlocks with large messages
    increaseSocketBuffers(network.get(), 128 * 1024 * 1024);

    json output_data;
    output_data["details"] = {{"num_parties", nP},
                              {"vec_size", vec_size},
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

    StatsPoint start(*network);

    network->sync();

    auto circ = generateCircuit(nP, pid, vec_size).orderGatesByLevel();
    network->sync();

    std::cout << "--- Circuit ---" << std::endl;
    std::cout << circ << std::endl;
    
    std::unordered_map<common::utils::wire_t, int> input_pid_map;
    int input_owner = 1; // Start with party 1
    for (const auto& g : circ.gates_by_level[0]) {
        if (g->type == common::utils::GateType::kInp) {
            input_pid_map[g->out] = input_owner;
            // Cycle through parties for each vec_size block of inputs
            static size_t input_count = 0;
            input_count++;
            if (input_count % vec_size == 0) {
                input_owner = (input_owner % nP) + 1;
            }
        }
    }

    std::cout << "Starting preprocessing" << std::endl;
    StatsPoint preproc_start(*network);
    int latency_us = static_cast<int>(latency * 1000);  // Convert ms to microseconds
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed, latency_us);
    auto preproc = off_eval.run(input_pid_map);
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    StatsPoint preproc_end(*network);

    std::cout << "Setting inputs" << std::endl;
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed, latency_us, use_pking);
    
    std::unordered_map<common::utils::wire_t, Ring> inputs;
    std::vector<common::utils::wire_t> input_wires;
    input_wires.reserve(input_pid_map.size());
    for (const auto& [wire, owner] : input_pid_map) {
        if (owner == static_cast<int>(pid)) {
            input_wires.push_back(wire);
        }
    }
    std::sort(input_wires.begin(), input_wires.end());

    // Create deterministic test inputs for each party's instance
    std::vector<std::vector<Ring>> party_input_values(nP);
    for (int p = 0; p < nP; ++p) {
        party_input_values[p].resize(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            // Party p's input: [p*1000+1, p*1000+2, ..., p*1000+vec_size]
            party_input_values[p][i] = static_cast<Ring>(p * 1000 + i + 1);
        }
    }

    if (!input_wires.empty()) {
        std::cout << "\n=== SETTING TEST INPUTS ===" << std::endl;
        std::cout << "Party " << pid << " setting inputs for its PermAndSh instance(s):" << std::endl;
        
        // Determine which instances this party owns based on input_pid_map
        size_t wire_idx = 0;
        for (int owner = 1; owner <= nP; ++owner) {
            if (owner == static_cast<int>(pid)) {
                std::cout << "  Instance owned by Party " << owner << " (pid=" << pid << "): [";
                for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
                    inputs[input_wires[wire_idx]] = party_input_values[owner - 1][i];
                    if (i < 10) {
                        std::cout << party_input_values[owner - 1][i] << (i < vec_size - 1 ? ", " : "");
                    }
                }
                if (vec_size > 10) std::cout << "...";
                std::cout << "]" << std::endl;
            }
        }
        std::cout << "========================\n" << std::endl;
    }

    eval.setInputs(inputs);
    
    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }

    auto outputs = eval.getOutputs();

    std::cout << "\n=== PERMUTE AND SHARE RESULT ===" << std::endl;
    std::cout << "Party " << pid << " reconstructed outputs:" << std::endl;
    std::cout << "  Total number of outputs: " << outputs.size() << " (should be " << nP * vec_size << ")" << std::endl;
    
    // Split outputs into nP groups (one per gate instance/owner)
    std::vector<std::vector<Ring>> instance_outputs(nP);
    for (int owner = 0; owner < nP; ++owner) {
        instance_outputs[owner].resize(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            instance_outputs[owner][i] = outputs[owner * vec_size + i];
        }
    }
    
    // Validate each instance
    bool all_valid = true;
    for (int owner = 0; owner < nP; ++owner) {
        std::cout << "\n  Instance owned by Party " << (owner + 1) << ":" << std::endl;
        std::cout << "    Input (first 20):  [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(20), party_input_values[owner].size()); ++i) {
            std::cout << party_input_values[owner][i] << (i + 1 == std::min(static_cast<size_t>(20), party_input_values[owner].size()) ? "" : ", ");
        }
        if (party_input_values[owner].size() > 20) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        std::cout << "    Output (first 20): [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(20), instance_outputs[owner].size()); ++i) {
            std::cout << instance_outputs[owner][i] << (i + 1 == std::min(static_cast<size_t>(20), instance_outputs[owner].size()) ? "" : ", ");
        }
        if (instance_outputs[owner].size() > 20) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        // Verify output is a permutation of input
        std::vector<Ring> sorted_inputs = party_input_values[owner];
        std::vector<Ring> sorted_outputs = instance_outputs[owner];
        std::sort(sorted_inputs.begin(), sorted_inputs.end());
        std::sort(sorted_outputs.begin(), sorted_outputs.end());
        
        bool is_valid_permutation = (sorted_inputs == sorted_outputs);
        if (is_valid_permutation) {
            std::cout << "    ✓ Valid permutation" << std::endl;
        } else {
            std::cout << "    ✗ INVALID permutation" << std::endl;
            all_valid = false;
        }
    }
    
    std::cout << "\n  " << (all_valid ? "✓ PERMUTE AND SHARE CORRECT" : "✗ PERMUTE AND SHARE INCORRECT") << std::endl;
    std::cout << "  All " << nP << " gate instances produced valid permutations" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    network->sync();
    StatsPoint online_end(*network);
    std::cout << "Online evaluation complete" << std::endl;

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
        ("vec-size,v", bpo::value<size_t>()->required(), "Number of elements in the vector.")
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
    bpo::options_description cmdline("Benchmark online phase for permute and share gates.");
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

// usage ./../run.sh permAndSh -n 2 -v 5 
