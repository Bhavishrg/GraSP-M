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

    std::cout << "Generating circuit with vec_size=" << vec_size << std::endl;
    
    common::utils::Circuit<Field> circ;

    // Create three sets of inputs for cPermAndSh gate
    // - input_vector: main inputs
    // - commitment_vector: values to add to input
    // - permuted_commitment_vector: values to subtract from output (will be set to pi(commitment_vector))
    std::vector<common::utils::wire_t> input_vector(vec_size);
    std::vector<common::utils::wire_t> commitment_vector(vec_size);
    std::vector<common::utils::wire_t> permuted_commitment_vector(vec_size);
    
    std::generate(input_vector.begin(), input_vector.end(), [&]() { return circ.newInputWire(); });
    std::generate(commitment_vector.begin(), commitment_vector.end(), [&]() { return circ.newInputWire(); });
    std::generate(permuted_commitment_vector.begin(), permuted_commitment_vector.end(), [&]() { return circ.newInputWire(); });

    // Create permutation for owner (party 1)
    std::vector<std::vector<int>> permutations(nP);
    
    for (int p = 0; p < nP; ++p) {
        permutations[p].resize(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            permutations[p][i] = i;
        }
        // Reverse permutation for testing
        std::reverse(permutations[p].begin(), permutations[p].end());
    }
    
    // Owner is party 1
    int owner = 1;

    // cPermAndSh gate: output = pi(input + commitment_vector) - permuted_commitment_vector
    // For correctness, we should have permuted_commitment_vector = pi(commitment_vector)
    // This way: output = pi(input + commitment_vector) - pi(commitment_vector) = pi(input)
    auto outputs = circ.addCPermAndShGate(input_vector, commitment_vector, permuted_commitment_vector, permutations, owner);

    // Set all outputs as circuit outputs
    for (const auto& out : outputs) {
        circ.setAsOutput(out);
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
    
    // Define the permutation for verification (must match circuit generation)
    std::vector<int> test_permutation(vec_size);
    for (int i = 0; i < vec_size; ++i) {
        test_permutation[i] = vec_size - 1 - i;  // Reverse permutation
    }
    
    // Set test values:
    // - First vec_size values (input_vector): 1, 2, 3, ..., vec_size
    // - Next vec_size values (commitment_vector): 100, 200, 300, ..., vec_size*100
    // - Last vec_size values (permuted_commitment_vector): pi(commitment_vector)
    std::vector<Field> input_vals(vec_size);
    std::vector<Field> comm_vals(vec_size);
    std::vector<Field> perm_comm_vals(vec_size);
    
    for (int i = 0; i < vec_size; ++i) {
        input_vals[i] = NTL::conv<Field>(i + 1);
        comm_vals[i] = NTL::conv<Field>((i + 1) * 100);
    }
    
    // Compute permuted_commitment_vector = pi(commitment_vector)
    for (int i = 0; i < vec_size; ++i) {
        perm_comm_vals[i] = comm_vals[test_permutation[i]];
    }
    
    // Set inputs in the circuit
    for (size_t i = 0; i < input_wires.size(); ++i) {
        auto wire = input_wires[i];
        if (i < static_cast<size_t>(vec_size)) {
            inputs[wire] = input_vals[i];
            std::cout << "  Wire " << wire << " (input[" << i << "]) = " << input_vals[i] << std::endl;
        } else if (i < static_cast<size_t>(2 * vec_size)) {
            size_t idx = i - vec_size;
            inputs[wire] = comm_vals[idx];
            std::cout << "  Wire " << wire << " (commitment_vector[" << idx << "]) = " << comm_vals[idx] << std::endl;
        } else {
            size_t idx = i - 2 * vec_size;
            inputs[wire] = perm_comm_vals[idx];
            std::cout << "  Wire " << wire << " (permuted_commitment_vector[" << idx << "]) = " << perm_comm_vals[idx] 
                      << " = pi(commitment_vector[" << test_permutation[idx] << "])" << std::endl;
        }
    }
    
    std::cout << "\n--- Expected Output ---" << std::endl;
    std::cout << "Since permuted_commitment_vector = pi(commitment_vector)," << std::endl;
    std::cout << "output = pi(input + commitment_vector) - permuted_commitment_vector" << std::endl;
    std::cout << "       = pi(input + commitment_vector) - pi(commitment_vector)" << std::endl;
    std::cout << "       = pi(input)" << std::endl;
    std::cout << "\nExpected outputs (pi(input) with reverse permutation):" << std::endl;
    for (int i = 0; i < vec_size; ++i) {
        Field expected = input_vals[test_permutation[i]];
        std::cout << "  Output[" << i << "] = input[" << test_permutation[i] << "] = " << expected << std::endl;
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
    std::cout << "\n=== CPERM AND SHARE RESULT ===" << std::endl;
    std::cout << "Party " << pid << " output:" << std::endl;
    std::cout << "  Number of outputs: " << outputs.size() << std::endl;
    std::cout << "  Expected: " << vec_size << " outputs" << std::endl;
    
    if (outputs.size() >= static_cast<size_t>(vec_size)) {
        std::cout << "\n  cPermAndSh gate outputs:" << std::endl;
        bool all_correct = true;
        for (int i = 0; i < vec_size; ++i) {
            Field expected = input_vals[test_permutation[i]];
            Field actual = outputs[i];
            bool correct = (expected == actual);
            all_correct = all_correct && correct;
            std::cout << "    Output[" << i << "] = " << actual 
                      << " (expected: " << expected << ")" 
                      << (correct ? " ✓" : " ✗") << std::endl;
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
        ("vec-size", bpo::value<int>()->default_value(4), "Size of vectors for cPermAndSh gate.");
  return desc;
}
// clang-format on

int main(int argc, char* argv[]) {
    auto prog_opts(programOptions());
    bpo::options_description cmdline("Benchmark online phase for cPermAndSh gates.");
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
// usage: ./../run.sh cpermsh --num-parties 3 --vec-size 4
