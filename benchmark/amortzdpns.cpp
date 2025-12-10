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

common::utils::Circuit<Field> generateCircuit(int nP, int pid, int vec_size) {

    std::cout << "Generating circuit with nP=" << nP << ", vec_size=" << vec_size << std::endl;
    
    common::utils::Circuit<Field> circ;

    // Create two separate sets of inputs for two parallel amortzdPnS gates
    std::vector<common::utils::wire_t> input_vector1(vec_size);
    std::vector<common::utils::wire_t> input_vector2(vec_size);
    
    std::generate(input_vector1.begin(), input_vector1.end(), [&]() { return circ.newInputWire(); });
    std::generate(input_vector2.begin(), input_vector2.end(), [&]() { return circ.newInputWire(); });

    // Create permutations for each party (nP permutations in total)
    // Indexed 0 to nP-1 (0-based indexing for the permutation vector)
    std::vector<std::vector<int>> permutations1(nP);
    std::vector<std::vector<int>> permutations2(nP);
    
    for (int p = 0; p < nP; ++p) {
        permutations1[p].resize(vec_size);
        permutations2[p].resize(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            permutations1[p][i] = i;
            permutations2[p][i] = i;
        }
        // Different permutations per party
        if (p == 0) {
            // Party 1 (index 0): reverse order
            std::reverse(permutations1[p].begin(), permutations1[p].end());
            std::reverse(permutations2[p].begin(), permutations2[p].end());
        } else if (p == 1 && vec_size > 1) {
            // Party 2 (index 1): rotate by 1
            std::rotate(permutations1[p].begin(), permutations1[p].begin() + 1, permutations1[p].end());
            std::rotate(permutations2[p].begin(), permutations2[p].begin() + 1, permutations2[p].end());
        } else if (p == 2 && vec_size > 1) {
            // Party 3 (index 2): rotate by 2
            int rot_amt = std::min(2, vec_size - 1);
            std::rotate(permutations1[p].begin(), permutations1[p].begin() + rot_amt, permutations1[p].end());
            std::rotate(permutations2[p].begin(), permutations2[p].begin() + rot_amt, permutations2[p].end());
        }
        // Other parties use identity permutation
    }

    // First amortzdPnS gate
    auto outputs1 = circ.addMOGate(common::utils::GateType::kAmortzdPnS, input_vector1, permutations1);
    
    // Second amortzdPnS gate (in parallel)
    auto outputs2 = circ.addMOGate(common::utils::GateType::kAmortzdPnS, input_vector2, permutations2);

    // Set all outputs from both gates as circuit outputs
    // outputs1 and outputs2 are 2D vectors: [party][wire_index]
    for (const auto& party_outputs : outputs1) {
        for (const auto& wire : party_outputs) {
            circ.setAsOutput(wire);
        }
    }
    for (const auto& party_outputs : outputs2) {
        for (const auto& wire : party_outputs) {
            circ.setAsOutput(wire);
        }
    }

    std::cout << "Circuit generated with " << outputs1.size() << " outputs" << std::endl;
    std::cout << "Expected: " << (nP * vec_size * 2) << " outputs (nP * vec_size per gate, 2 gates)" << std::endl;

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
    auto latency = opts["latency"].as<double>();
    auto pid = opts["pid"].as<size_t>();
    auto threads = opts["threads"].as<size_t>();
    auto seed = opts["seed"].as<size_t>();
    auto repeat = opts["repeat"].as<size_t>();
    auto port = opts["port"].as<int>();
    auto use_pking = opts["use-pking"].as<bool>();
    auto vec_size = opts["vec-size"].as<int>();


    // Initialize NTL library with proper modulus and thread settings
    initNTL(threads);

    omp_set_nested(1);
    if (nP < 10) { omp_set_num_threads(nP); }
    else { omp_set_num_threads(10); }

    std::cout << "Starting benchmarks" << std::endl;

    std::string net_config = opts.count("net-config") ? opts["net-config"].as<std::string>() : "";
    std::shared_ptr<io::NetIOMP> network = createNetwork(pid, nP, latency, port,
                                                          opts["localhost"].as<bool>(),
                                                          net_config);

    // Increase socket buffer sizes to prevent deadlocks with large messages
    increaseSocketBuffers(network.get(), 128 * 1024 * 1024);

    json output_data;
    output_data["details"] = {{"num_parties", nP},
                              {"latency (ms)", latency},
                              {"pid", pid},
                              {"threads", threads},
                              {"seed", seed},
                              {"repeat", repeat},
                              {"vec_size", vec_size}};
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
    for (const auto& g : circ.gates_by_level[0]) {
        if (g->type == common::utils::GateType::kInp) {
            input_pid_map[g->out] = 1;  // All inputs owned by party 1
        }
    }

    std::cout << "Starting preprocessing" << std::endl;
    StatsPoint preproc_start(*network);
    int latency_us = static_cast<int>(latency * 1000);  // Convert ms to microseconds
    
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed, latency_us, use_pking);
    auto preproc = off_eval.run(input_pid_map);
    
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    
    StatsPoint preproc_end(*network);

    std::cout << "Setting inputs" << std::endl;
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed, latency_us, use_pking);
    
    // Set specific test inputs
    std::unordered_map<common::utils::wire_t, Field> inputs;
    
    // Collect all input wires owned by this party
    std::vector<common::utils::wire_t> input_wires;
    for (const auto& [wire, owner] : input_pid_map) {
        if (owner == static_cast<int>(pid)) {
            input_wires.push_back(wire);
        }
    }
    
    // Sort to ensure consistent ordering
    std::sort(input_wires.begin(), input_wires.end());
    
    std::cout << "\n=== SETTING TEST INPUTS ===" << std::endl;
    std::cout << "Party " << pid << " setting inputs:" << std::endl;
    
    // Set test values: first vec_size values are 1, 2, 3, ..., vec_size
    // Next vec_size values are 10, 20, 30, ..., vec_size*10
    for (size_t i = 0; i < input_wires.size(); ++i) {
        auto wire = input_wires[i];
        if (i < static_cast<size_t>(vec_size)) {
            Field val = NTL::conv<Field>(i + 1);
            inputs[wire] = val;
            std::cout << "  Wire " << wire << " (input1[" << i << "]) = " << val << std::endl;
        } else {
            Field val = NTL::conv<Field>((i - vec_size + 1) * 10);
            inputs[wire] = val;
            std::cout << "  Wire " << wire << " (input2[" << (i - vec_size) << "]) = " << val << std::endl;
        }
    }
    std::cout << "========================\n" << std::endl;
    
    // Set inputs in the evaluator
    std::cout << "Setting inputs in evaluator" << std::endl;
    eval.setInputs(inputs);
    
    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }
    
    // Get and print outputs
    auto outputs = eval.getOutputs();
    std::cout << "\n=== AMORTIZED PERM-AND-SHARE (AmortzdPnS) RESULT ===" << std::endl;
    std::cout << "Party " << pid << " output:" << std::endl;
    std::cout << "  Number of outputs: " << outputs.size() << std::endl;
    std::cout << "  Expected: " << (nP * vec_size * 2) << " outputs (2 amortzdPnS gates, each produces nP*vec_size outputs)" << std::endl;
    
    if (outputs.size() >= static_cast<size_t>(nP * vec_size * 2)) {
        std::cout << "\n  First amortzdPnS gate outputs:" << std::endl;
        for (int party = 1; party <= nP; ++party) {
            std::cout << "    Party " << party << "'s permuted shares:" << std::endl;
            for (int i = 0; i < vec_size; ++i) {
                int idx = (party - 1) * vec_size + i;
                std::cout << "      Output[" << idx << "] = " << outputs[idx] << std::endl;
            }
        }
        
        std::cout << "\n  Second amortzdPnS gate outputs:" << std::endl;
        int offset = nP * vec_size;
        for (int party = 1; party <= nP; ++party) {
            std::cout << "    Party " << party << "'s permuted shares:" << std::endl;
            for (int i = 0; i < vec_size; ++i) {
                int idx = offset + (party - 1) * vec_size + i;
                std::cout << "      Output[" << idx << "] = " << outputs[idx] << std::endl;
            }
        }
    }
    std::cout << "================================================\n" << std::endl;
    
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
        ("latency,l", bpo::value<double>()->default_value(0.5), "Network latency in ms.")
        ("pid,p", bpo::value<size_t>()->required(), "Party ID.")
        ("threads,t", bpo::value<size_t>()->default_value(6), "Number of threads (recommended 6).")
        ("seed", bpo::value<size_t>()->default_value(200), "Value of the random seed.")
        ("net-config", bpo::value<std::string>(), "Path to JSON file containing network details of all parties.")
        ("localhost", bpo::bool_switch(), "All parties are on same machine.")
        ("port", bpo::value<int>()->default_value(10000), "Base port for networking.")
        ("output,o", bpo::value<std::string>(), "File to save benchmarks.")
        ("repeat,r", bpo::value<size_t>()->default_value(1), "Number of times to run benchmarks.")
        ("use-pking", bpo::value<bool>()->default_value(true), "Use king party for reconstruction (true) or direct reconstruction (false).")
        ("vec-size", bpo::value<int>()->default_value(4), "Size of vectors for amortzdPnS gates.");
  return desc;
}
// clang-format on

int main(int argc, char* argv[]) {
    auto prog_opts(programOptions());
    bpo::options_description cmdline("Benchmark online phase for amortzdPnS gates.");
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
// usage: ./../run.sh amortzdpns --num-parties 3 --vec-size 4
