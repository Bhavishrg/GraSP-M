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

common::utils::Circuit<Ring> generateCompactionCircuit(int nP, int pid, size_t vec_size) {

    std::cout << "Generating compaction circuit" << std::endl;
    
    common::utils::Circuit<Ring> circ;

    // Input: vector t (binary: 0 or 1) and vector p (payloads)
    std::vector<common::utils::wire_t> t_vector(vec_size);
    std::vector<common::utils::wire_t> p_vector(vec_size);
    
    std::generate(t_vector.begin(), t_vector.end(), [&]() { return circ.newInputWire(); });
    std::generate(p_vector.begin(), p_vector.end(), [&]() { return circ.newInputWire(); });

    // Step 1: Construct c0 and c1 vectors to count number of 0s and 1s
    std::vector<common::utils::wire_t> c1_vector(vec_size);
    std::vector<common::utils::wire_t> c0_vector(vec_size);
    
    // c1[0] = t[0]
    c1_vector[0] = t_vector[0];
    
    // c0[0] = 1 - c1[0]
    c0_vector[0] = circ.addConstOpGate(common::utils::GateType::kConstAdd, 
                                       circ.addConstOpGate(common::utils::GateType::kConstMul, c1_vector[0], -1), 
                                       1);
    
    // For j = 1 to N-1:
    for (size_t j = 1; j < vec_size; ++j) {
        // c1[j] = c1[j-1] + t[j]
        c1_vector[j] = circ.addGate(common::utils::GateType::kAdd, c1_vector[j-1], t_vector[j]);
        c0_vector[j] = circ.addConstOpGate(common::utils::GateType::kConstAdd, 
                                       circ.addConstOpGate(common::utils::GateType::kConstMul, c1_vector[j], -1), 
                                       j+1);
    }

    
    
    // Step 2: Construct label vector
    // label[j] = { c0[j] + c1[N-1], if t[j] = 0
    //            { c1[j],           otherwise
    // 
    // We implement this as: label[j] = (c0[j] + c1[N-1] - c1[j])(1 - t[j]) + c1[j]
    
    std::vector<common::utils::wire_t> label_vector(vec_size);
    auto c1_last = c1_vector[vec_size - 1];

    std::vector<common::utils::wire_t> label_reconstructed1(vec_size);
    std::vector<common::utils::wire_t> label_reconstructed2(vec_size);
    std::vector<common::utils::wire_t> label_reconstructed3(vec_size);

    for (size_t j = 0; j < vec_size; ++j) {
        label_reconstructed1[j] = circ.addGate(common::utils::GateType::kRec, c1_vector[j]);
    }
    
    for (size_t j = 0; j < vec_size; ++j) {
        // Compute c0[j] + c1[N-1]
        auto c0_plus_c1last = circ.addGate(common::utils::GateType::kAdd, c0_vector[j], c1_last);
        
        // Compute c0[j] + c1[N-1] - c1[j]
        auto diff_term = circ.addGate(common::utils::GateType::kSub, c0_plus_c1last, c1_vector[j]);

        // Compute (1 - t[j]) = 1 + (-1)*t[j]
        auto one_minus_t = circ.addConstOpGate(common::utils::GateType::kConstMul, t_vector[j], -1);
        one_minus_t = circ.addConstOpGate(common::utils::GateType::kConstAdd, one_minus_t, 1);

        
        // Compute (c0[j] + c1[N-1] - c1[j]) * (1 - t[j])
        auto mult_term = circ.addGate(common::utils::GateType::kMul, diff_term, one_minus_t);
        label_reconstructed2[j] = circ.addGate(common::utils::GateType::kRec, mult_term);

        // label[j] = (c0[j] + c1[N-1] - c1[j]) * (1 - t[j]) + c1[j]
        label_vector[j] = circ.addGate(common::utils::GateType::kAdd, mult_term, c1_vector[j]);
        label_reconstructed3[j] = circ.addGate(common::utils::GateType::kRec, label_vector[j]);
        
    }
    
    // Step 3: Shuffle the elements in p, t, label using the same random permutation
    // Generate permutation
    // std::vector<std::vector<int>> permutation;
    // std::vector<int> tmp_perm(vec_size);
    // for (size_t i = 0; i < vec_size; ++i) {
    //     tmp_perm[i] = i;
    // }
    // permutation.push_back(tmp_perm);
    // if (pid == 0) {
    //     for (int i = 1; i < nP; ++i) {
    //         permutation.push_back(tmp_perm);
    //     }
    // }
    
    // auto p_shuffled = circ.addMGate(common::utils::GateType::kShuffle, p_vector, permutation);
    // auto t_shuffled = circ.addMGate(common::utils::GateType::kShuffle, t_vector, permutation);
    // auto label_shuffled = circ.addMGate(common::utils::GateType::kShuffle, label_vector, permutation);
    
    // Step 4: Reconstruct label to get the actual permutation values
    // Use reconstruction gate to reveal label values
    
    // std::vector<common::utils::wire_t> label_reconstructed(vec_size);
    // for (size_t i = 0; i < vec_size; ++i) {
    //     // label_reconstructed[i] = circ.addGate(common::utils::GateType::kRec, label_shuffled[i]);
    //     label_reconstructed[i] = circ.addGate(common::utils::GateType::kRec, label_vector[i]);
    // }
    
    // Step 5: Apply permutation using kPublicPerm
    // The permutation will be determined at runtime from reconstructed labels
    // We encode the wire IDs of label_reconstructed in the permutation field using negative values
    // as a signal that these should be read from wires during evaluation
    //
    // Format: permutation[i] = -(label_reconstructed[i] + 1)
    // This allows the evaluator to detect (perm < 0) and read from wire (-perm - 1)
    
    // std::vector<int> dynamic_perm(vec_size);
    // for (size_t i = 0; i < vec_size; ++i) {
    //     // Encode: negative value means "read from this wire ID"
    //     // We use -(wire_id + 1) to avoid confusion with wire 0
    //     dynamic_perm[i] = -(static_cast<int>(label_reconstructed[i]) + 1);
    // }
    
    // auto p_compacted = circ.addConstOpMGate(common::utils::GateType::kPublicPerm, p_shuffled, dynamic_perm);
    // auto t_compacted = circ.addConstOpMGate(common::utils::GateType::kPublicPerm, t_shuffled, dynamic_perm);
    
    // Set outputs: compacted t and p vectors
    // for (size_t i = 0; i < vec_size; ++i) {
    //     circ.setAsOutput(t_compacted[i]);
    //     circ.setAsOutput(p_compacted[i]);
    // }

    // Alternative: Output original t and p vectors for verification        
    for (size_t i = 0; i < vec_size; ++i) {
        circ.setAsOutput(t_vector[i]);
        circ.setAsOutput(p_vector[i]);
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

    auto circ = generateCompactionCircuit(nP, pid, vec_size).orderGatesByLevel();
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
    // emp::PRG prg(&emp::zero_block, seed);
    int latency_us = static_cast<int>(latency * 1000);  // Convert ms to microseconds
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed, latency_us);
    auto preproc = off_eval.run(input_pid_map);
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    StatsPoint preproc_end(*network);

    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed, latency_us);
    
    // Set inputs: binary values for t vector (first vec_size wires) and payload values for p vector (second vec_size wires)
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
    
    // First vec_size wires are for t vector (binary: 0 or 1)
    // Second vec_size wires are for p vector (payload values)
    for (size_t i = 0; i < input_wires.size(); ++i) {
        auto wire = input_wires[i];
        if (i < vec_size) {
            // t vector: binary values (0 or 1)
            inputs[wire] = static_cast<Ring>(rand() % 2);
        } else {
            // p vector: random payload values
            inputs[wire] = static_cast<Ring>(rand() % 100);
        }
    }

    // Print first few inputs for verification
    for (size_t i = 0; i < std::min(size_t(20), inputs.size()); ++i) {
        std::cout << "Input wire " << input_wires[i] << " (index " << i << "): " << inputs[input_wires[i]] << std::endl;
    }
    std::cout << "Total inputs set: " << inputs.size() << std::endl;
    
    eval.setInputs(inputs);
    
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }
    
    auto outputs = eval.getOutputs();
    std::cout << "Number of outputs: " << outputs.size() << std::endl;
    std::cout << "First few outputs: ";
    for (size_t i = 0; i < std::min(size_t(20), outputs.size()); ++i) {
        std::cout << outputs[i] << " ";
    }
    std::cout << std::endl;
    
    network->sync();
    StatsPoint online_end(*network);

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
        ("vec-size,v", bpo::value<size_t>()->required(), "Size of the vector to compact.")
        ("latency,l", bpo::value<double>()->required(), "Network latency in ms.")
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
