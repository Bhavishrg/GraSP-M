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

common::utils::Circuit<Ring> generateCircuit(int nP, int pid, size_t vec_size, size_t num_payloads) {

    std::cout << "Generating circuit with vec_size=" << vec_size 
              << ", num_payloads=" << num_payloads << std::endl;

    common::utils::Circuit<Ring> circ;

    // Create position map input wires (secret shares)
    std::vector<common::utils::wire_t> position_map_shares(vec_size);
    std::generate(position_map_shares.begin(), position_map_shares.end(), [&]() { return circ.newInputWire(); });

    // Reconstruct position map to get public permutation values
    std::vector<common::utils::wire_t> position_map_reconstructed(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        position_map_reconstructed[i] = circ.addGate(common::utils::GateType::kRec, position_map_shares[i]);
    }

    // Create payload input wires
    std::vector<std::vector<common::utils::wire_t>> payloads(num_payloads);
    for (size_t p = 0; p < num_payloads; ++p) {
        payloads[p].resize(vec_size);
        std::generate(payloads[p].begin(), payloads[p].end(), [&]() { return circ.newInputWire(); });
    }

    // Add rewire gate with reconstructed position map
    auto outputs = circ.addRewireGate(position_map_reconstructed, payloads);

    // Set outputs
    for (size_t p = 0; p < num_payloads; ++p) {
        for (size_t i = 0; i < vec_size; ++i) {
            circ.setAsOutput(outputs[p][i]);
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
    auto num_payloads = opts["num-payloads"].as<size_t>();
    auto iter = opts["iter"].as<int>();
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
    increaseSocketBuffers(network.get(), 128 * 1024 * 1024);

    json output_data;
    output_data["details"] = {{"num_parties", nP},
                              {"vec_size", vec_size},
                              {"num_payloads", num_payloads},
                              {"iterations", iter},
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

    auto circ = generateCircuit(nP, pid, vec_size, num_payloads).orderGatesByLevel();
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
    
    std::unordered_map<common::utils::wire_t, Ring> inputs;
    std::vector<common::utils::wire_t> input_wires;
    input_wires.reserve(input_pid_map.size());
    for (const auto& [wire, owner] : input_pid_map) {
        if (owner == static_cast<int>(pid)) {
            input_wires.push_back(wire);
        }
    }
    std::sort(input_wires.begin(), input_wires.end());

    if (!input_wires.empty()) {
        std::cout << "\n=== SETTING TEST INPUTS ===" << std::endl;
        std::cout << "Party " << pid << " setting inputs:" << std::endl;
        
        // Create a test permutation (reverse order for demonstration)
        std::vector<Ring> position_map_values(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            position_map_values[i] = static_cast<Ring>(vec_size - 1 - i);  // Reverse permutation
        }
        
        // Create test payload data
        std::vector<std::vector<Ring>> payload_values(num_payloads);
        for (size_t p = 0; p < num_payloads; ++p) {
            payload_values[p].resize(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                payload_values[p][i] = static_cast<Ring>(p * 1000 + i + 1);  // Distinct values per payload
            }
        }
        
        // Set position map inputs
        for (size_t idx = 0; idx < vec_size && idx < input_wires.size(); ++idx) {
            inputs[input_wires[idx]] = position_map_values[idx];
        }
        
        // Set payload inputs
        size_t wire_idx = vec_size;
        for (size_t p = 0; p < num_payloads; ++p) {
            for (size_t i = 0; i < vec_size && wire_idx < input_wires.size(); ++i, ++wire_idx) {
                inputs[input_wires[wire_idx]] = payload_values[p][i];
            }
        }
        
        std::cout << "  Position map (first 10): [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), vec_size); ++i) {
            std::cout << position_map_values[i] << (i + 1 == std::min(static_cast<size_t>(10), vec_size) ? "" : ", ");
        }
        if (vec_size > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        for (size_t p = 0; p < std::min(static_cast<size_t>(3), num_payloads); ++p) {
            std::cout << "  Payload " << p << " (first 10): [";
            for (size_t i = 0; i < std::min(static_cast<size_t>(10), vec_size); ++i) {
                std::cout << payload_values[p][i] << (i + 1 == std::min(static_cast<size_t>(10), vec_size) ? "" : ", ");
            }
            if (vec_size > 10) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        
        std::cout << "  Set " << input_wires.size() << " input values" << std::endl;
        std::cout << "========================\n" << std::endl;
    }

    eval.setInputs(inputs);
    
    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }

    auto outputs = eval.getOutputs();

    std::cout << "\n=== REWIRE RESULT ===" << std::endl;
    std::cout << "Party " << pid << " reconstructed outputs:" << std::endl;
    std::cout << "  Total number of outputs: " << outputs.size() << std::endl;
    std::cout << "  Outputs per payload: " << vec_size << std::endl;
    
    // Display rewired outputs
    for (size_t p = 0; p < std::min(static_cast<size_t>(3), num_payloads); ++p) {
        std::cout << "  Payload " << p << " output (first 10): [";
        size_t start_idx = p * vec_size;
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), vec_size) && (start_idx + i) < outputs.size(); ++i) {
            std::cout << outputs[start_idx + i] << (i + 1 == std::min(static_cast<size_t>(10), vec_size) ? "" : ", ");
        }
        if (vec_size > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  ✓ REWIRE COMPLETE - Payloads permuted according to position map" << std::endl;
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
        ("vec-size,v", bpo::value<size_t>()->required(), "Size of position map and payload vectors.")
        ("num-payloads", bpo::value<size_t>()->default_value(2), "Number of payload vectors to rewire.")
        ("iter,i", bpo::value<int>()->default_value(1), "Number of iterations for message passing.")
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
    bpo::options_description cmdline("Benchmark online phase for rewire gates.");
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
