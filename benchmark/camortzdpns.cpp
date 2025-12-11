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

    // Create inputs for cAmortzdPnS gate:
    // - input_vector: main inputs
    // - commitment_vector: values to add to input
    // - permuted_commitment_vectors[i]: values to subtract from output i (should equal pi(commitment_vector))
    std::vector<common::utils::wire_t> input_vector(vec_size);
    std::vector<common::utils::wire_t> commitment_vector(vec_size);
    std::vector<std::vector<common::utils::wire_t>> permuted_commitment_vectors(nP, std::vector<common::utils::wire_t>(vec_size));
    
    std::generate(input_vector.begin(), input_vector.end(), [&]() { return circ.newInputWire(); });
    std::generate(commitment_vector.begin(), commitment_vector.end(), [&]() { return circ.newInputWire(); });
    for (int p = 0; p < nP; ++p) {
        std::generate(permuted_commitment_vectors[p].begin(), permuted_commitment_vectors[p].end(), 
                      [&]() { return circ.newInputWire(); });
    }

    // Create permutations for each party (nP permutations in total)
    std::vector<std::vector<int>> permutations(nP);
    
    for (int p = 0; p < nP; ++p) {
        permutations[p].resize(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            permutations[p][i] = i;
        }
        // Different permutations per party
        if (p == 0) {
            // Party 1 (index 0): reverse order
            std::reverse(permutations[p].begin(), permutations[p].end());
        } else if (p == 1 && vec_size > 1) {
            // Party 2 (index 1): rotate by 1
            std::rotate(permutations[p].begin(), permutations[p].begin() + 1, permutations[p].end());
        } else if (p == 2 && vec_size > 1) {
            // Party 3 (index 2): rotate by 2
            int rot_amt = std::min(2, vec_size - 1);
            std::rotate(permutations[p].begin(), permutations[p].begin() + rot_amt, permutations[p].end());
        }
        // Other parties use identity permutation
    }

    // cAmortzdPnS gate: output[i] = pi(input + commitment) - permuted_commitment[i]
    // For correctness: permuted_commitment[i] = pi(commitment)
    // This gives: output[i] = pi(input + commitment) - pi(commitment) = pi(input)
    auto outputs = circ.addCAmortzdPnSGate(input_vector, commitment_vector, permuted_commitment_vectors, permutations);

    // Set all outputs as circuit outputs
    for (const auto& party_outputs : outputs) {
        for (const auto& wire : party_outputs) {
            circ.setAsOutput(wire);
        }
    }

    std::cout << "Circuit generated with " << outputs.size() * outputs[0].size() << " outputs" << std::endl;
    std::cout << "Expected: " << (nP * vec_size) << " outputs (nP * vec_size)" << std::endl;

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
    
    // Define the permutations for verification (must match circuit generation)
    std::vector<std::vector<int>> test_permutations(nP);
    for (int p = 0; p < nP; ++p) {
        test_permutations[p].resize(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            test_permutations[p][i] = i;
        }
        if (p == 0) {
            std::reverse(test_permutations[p].begin(), test_permutations[p].end());
        } else if (p == 1 && vec_size > 1) {
            std::rotate(test_permutations[p].begin(), test_permutations[p].begin() + 1, test_permutations[p].end());
        } else if (p == 2 && vec_size > 1) {
            int rot_amt = std::min(2, vec_size - 1);
            std::rotate(test_permutations[p].begin(), test_permutations[p].begin() + rot_amt, test_permutations[p].end());
        }
    }
    
    // Set test values:
    // - First vec_size values (input_vector): 1, 2, 3, ..., vec_size
    // - Next vec_size values (commitment_vector): 100, 200, 300, ..., vec_size*100
    // - Next nP*vec_size values (permuted_commitment_vectors): pi(commitment_vector) for each party
    std::vector<Field> input_vals(vec_size);
    std::vector<Field> comm_vals(vec_size);
    std::vector<std::vector<Field>> perm_comm_vals(nP, std::vector<Field>(vec_size));
    
    for (int i = 0; i < vec_size; ++i) {
        input_vals[i] = NTL::conv<Field>(i + 1);
        comm_vals[i] = NTL::conv<Field>((i + 1) * 100);
    }
    
    // Compute permuted_commitment_vectors = pi(commitment_vector) for each party
    for (int p = 0; p < nP; ++p) {
        for (int i = 0; i < vec_size; ++i) {
            perm_comm_vals[p][i] = comm_vals[test_permutations[p][i]];
        }
    }
    
    // Set inputs in the circuit
    size_t wire_idx = 0;
    for (size_t i = 0; i < input_wires.size(); ++i) {
        auto wire = input_wires[i];
        if (i < static_cast<size_t>(vec_size)) {
            // input_vector
            inputs[wire] = input_vals[i];
            std::cout << "  Wire " << wire << " (input[" << i << "]) = " << input_vals[i] << std::endl;
        } else if (i < static_cast<size_t>(2 * vec_size)) {
            // commitment_vector
            size_t idx = i - vec_size;
            inputs[wire] = comm_vals[idx];
            std::cout << "  Wire " << wire << " (commitment[" << idx << "]) = " << comm_vals[idx] << std::endl;
        } else {
            // permuted_commitment_vectors
            size_t offset = i - 2 * vec_size;
            int party_idx = offset / vec_size;
            size_t elem_idx = offset % vec_size;
            inputs[wire] = perm_comm_vals[party_idx][elem_idx];
            std::cout << "  Wire " << wire << " (perm_comm[" << party_idx << "][" << elem_idx << "]) = " 
                      << perm_comm_vals[party_idx][elem_idx] << " = pi_" << party_idx 
                      << "(commitment[" << test_permutations[party_idx][elem_idx] << "])" << std::endl;
        }
    }
    
    std::cout << "\n--- Expected Outputs ---" << std::endl;
    std::cout << "Since permuted_commitment[i] = pi(commitment) for each party i," << std::endl;
    std::cout << "output[i] = pi(input + commitment) - permuted_commitment[i]" << std::endl;
    std::cout << "          = pi(input + commitment) - pi(commitment)" << std::endl;
    std::cout << "          = pi(input)" << std::endl;
    std::cout << "\nExpected outputs for each party:" << std::endl;
    for (int p = 0; p < nP; ++p) {
        std::cout << "  Party " << (p + 1) << " (permutation " << p << "):" << std::endl;
        for (int i = 0; i < vec_size; ++i) {
            Field expected = input_vals[test_permutations[p][i]];
            std::cout << "    Output[" << i << "] = input[" << test_permutations[p][i] << "] = " << expected << std::endl;
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
    std::cout << "\n=== COMMITTED AMORTIZED PNS RESULT ===" << std::endl;
    std::cout << "Party " << pid << " output:" << std::endl;
    std::cout << "  Number of outputs: " << outputs.size() << std::endl;
    std::cout << "  Expected: " << (nP * vec_size) << " outputs (nP * vec_size)" << std::endl;
    
    if (outputs.size() >= static_cast<size_t>(nP * vec_size)) {
        std::cout << "\n  cAmortzdPnS gate outputs:" << std::endl;
        bool all_correct = true;
        for (int p = 0; p < nP; ++p) {
            std::cout << "  Party " << (p + 1) << " outputs:" << std::endl;
            for (int i = 0; i < vec_size; ++i) {
                size_t out_idx = p * vec_size + i;
                Field expected = input_vals[test_permutations[p][i]];
                Field actual = outputs[out_idx];
                bool correct = (expected == actual);
                all_correct = all_correct && correct;
                std::cout << "    Output[" << i << "] = " << actual 
                          << " (expected: " << expected << ")" 
                          << (correct ? " ✓" : " ✗") << std::endl;
            }
        }
        std::cout << "\n  Verification: " << (all_correct ? "PASSED ✓" : "FAILED ✗") << std::endl;
    }
    std::cout << "============================\n" << std::endl;
    
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
        ("vec-size", bpo::value<int>()->default_value(4), "Size of vectors for cAmortzdPnS gate.");
  return desc;
}
// clang-format on

int main(int argc, char* argv[]) {
    auto prog_opts(programOptions());
    bpo::options_description cmdline("Benchmark online phase for cAmortzdPnS gates.");
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
// usage: ./../run.sh camortzdpns --num-parties 3 --vec-size 4
