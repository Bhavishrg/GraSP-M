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

common::utils::Circuit<Ring> generateSortCircuit(int nP, int pid, size_t vec_size) {

    std::cout << "Generating sort circuit for " << vec_size << " elements (32-bit each)" << std::endl;
    
    common::utils::Circuit<Ring> circ;

    // Input: vec_size many 32-bit values, bit-decomposed
    // Total wires: vec_size * 32
    std::vector<common::utils::wire_t> input_bits(vec_size * 32);
    
    std::generate(input_bits.begin(), input_bits.end(), [&]() { return circ.newInputWire(); });

    // Generate identity permutation for preprocessing
    std::vector<std::vector<int>> permutations;
    std::vector<int> identity_perm(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        identity_perm[i] = i;
    }
    permutations.push_back(identity_perm);
    if (pid == 0) {
        for (int i = 1; i < nP; ++i) {
            permutations.push_back(identity_perm);
        }
    }

    // Use the sort gate
    auto sorted_bits = circ.addSortGate(input_bits, permutations);
    
    // Set outputs: all sorted bit-decomposed values
    for (size_t i = 0; i < vec_size * 32; ++i) {
        circ.setAsOutput(sorted_bits[i]);
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

    auto vec_size = opts["vec-size"].as<size_t>();
    auto nP = opts["num-parties"].as<int>();
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

    std::shared_ptr<io::NetIOMP> network = nullptr;
    if (opts["localhost"].as<bool>()) {
        network = std::make_shared<io::NetIOMP>(pid, nP + 1, latency, port, nullptr, true);
    } else {
        std::ifstream fnet(opts["net-config"].as<std::string>());
        if (!fnet.good()) {
            fnet.close();
            throw std::runtime_error("Could not open network config file");
        }
        json netdata;
        fnet >> netdata;
        fnet.close();
        std::vector<std::string> ipaddress(nP + 1);
        std::array<char*, 5> ip{};
        for (size_t i = 0; i < nP + 1; ++i) {
            ipaddress[i] = netdata[i].get<std::string>();
            ip[i] = ipaddress[i].data();
        }
        network = std::make_shared<io::NetIOMP>(pid, nP + 1, latency, port, ip.data(), false);
    }

    // Increase socket buffer sizes to prevent deadlocks with large messages
    // Calculate based on expected output size: vec_size * 32 bits * 8 bytes per Ring
    size_t expected_output_bytes = vec_size * 32 * sizeof(Ring);
    int buffer_size = std::max(128 * 1024 * 1024, 
                                static_cast<int>(expected_output_bytes * (nP - 1) * 3));
    increaseSocketBuffers(network.get(), buffer_size);

    json output_data;
    output_data["details"] = {{"num_parties", nP},
                              {"vec_size", vec_size},
                              {"total_bits", vec_size * 32},
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

    auto circ = generateSortCircuit(nP, pid, vec_size).orderGatesByLevel();
    network->sync();

    std::cout << "--- Circuit ---" << std::endl;
    std::cout << circ << std::endl;
    
    std::unordered_map<common::utils::wire_t, int> input_pid_map;
    for (const auto& g : circ.gates_by_level[0]) {
        if (g->type == common::utils::GateType::kInp) {
            input_pid_map[g->out] = 1;
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
    
    // Set inputs: 32-bit values, bit-decomposed
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
    
    // Store input values for verification (32-bit integers)
    std::vector<uint32_t> input_values(vec_size);
    
    // Generate random 32-bit values and decompose into bits
    for (size_t elem = 0; elem < vec_size; ++elem) {
        uint32_t value = static_cast<uint32_t>(rand()) % 1000; // Keep values small for easier verification
        input_values[elem] = value;
        
        // Decompose into 32 bits: wire order is [bit0, bit1, ..., bit31] for each element
        for (size_t bit = 0; bit < 32; ++bit) {
            size_t wire_idx = elem * 32 + bit;
            if (wire_idx < input_wires.size()) {
                Ring bit_value = static_cast<Ring>((value >> bit) & 1);
                inputs[input_wires[wire_idx]] = bit_value;
            }
        }
    }

    // Print first 10 input values
    std::cout << "\n=== INPUT VALUES (Party " << pid << ") ===" << std::endl;
    std::cout << "First 10 values: ";
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), vec_size); ++i) {
        std::cout << input_values[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Total inputs set: " << inputs.size() << " bits (" << vec_size << " values)" << std::endl;
    
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
    std::cout << "Number of output bits: " << outputs.size() << std::endl;
    
    // Reconstruct 32-bit values from output bits
    std::vector<uint32_t> output_values(vec_size);
    for (size_t elem = 0; elem < vec_size; ++elem) {
        uint32_t value = 0;
        for (size_t bit = 0; bit < 32; ++bit) {
            size_t output_idx = elem * 32 + bit;
            if (output_idx < outputs.size()) {
                uint32_t bit_value = static_cast<uint32_t>(outputs[output_idx]) & 1;
                value |= (bit_value << bit);
            }
        }
        output_values[elem] = value;
    }
    
    // Print first 10 output values
    std::cout << "\n=== OUTPUT VALUES (Party " << pid << ") ===" << std::endl;
    std::cout << "First 10 sorted values: ";
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), vec_size); ++i) {
        std::cout << output_values[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify sorting (check if output is sorted)
    bool is_sorted = true;
    for (size_t i = 1; i < vec_size; ++i) {
        if (output_values[i] < output_values[i-1]) {
            is_sorted = false;
            break;
        }
    }
    std::cout << "Sorting verification: " << (is_sorted ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
    
    

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
                            {"peak_resident_set_size", peakResidentSetSize()},
                            {"sorting_verified", is_sorted}};

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
        ("vec-size,v", bpo::value<size_t>()->required(), "Number of 32-bit values to sort.")
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
    bpo::options_description cmdline("Benchmark secure sort circuit.");
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
