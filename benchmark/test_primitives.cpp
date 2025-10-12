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

common::utils::Circuit<Ring> generateCircuit(std::shared_ptr<io::NetIOMP> &network, int nP, int pid, size_t vec_size, int iter) {

    std::cout << "Generating circuit" << std::endl;

    network->send(0, &nP, sizeof(int));

    std::vector<std::vector<int>> permutation;
    std::vector<int> tmp_perm(vec_size);
    for (int i = 0; i < vec_size; ++i) {
        tmp_perm[i] = i;
    }
    permutation.push_back(tmp_perm);
    if (pid == 0) {
        for (int i = 1; i < nP; ++i) {
            permutation.push_back(tmp_perm);
        }
    }

    common::utils::Circuit<Ring> circ;
    int in_size = vec_size;
    for (int p = 0; p < nP; ++p) {
        std::vector<wire_t> input(in_size);
        for (int i = 0; i < input.size(); ++i) {
            input[i] = circ.newInputWire();
            // std::cout << "in " << input[i] << std::endl;
        }
        auto out = circ.addConstOpMGate(common::utils::GateType::kPublicPerm, input, tmp_perm);
        for (int i = 0; i < out.size(); ++i) {
            // std::cout << "out " << out[i] << std::endl;
            circ.setAsOutput(out[i]);
        }
    }

    // auto mul = circ.addGate(common::utils::GateType::kMul, input[0], input[1]);
    
    // std::vector<wire_t> input(vec_size);
    // for (int i = 0; i < vec_size; ++i) {
    //     input[i] = circ.newInputWire();
    // }
    // auto out = circ.addMOGate(common::utils::GateType::kAmortzdPnS, input, nP);
    // for (int i = 0; i < out.size(); ++i) {
    //     for (int j = 0; j < out[i].size(); ++j) {
    //         circ.setAsOutput(out[i][j]);
    //     }
    // }

    // for (int i = 0; i < nP; ++i) {
        // auto input1 = circ.newInputWire();
        // auto input2 = circ.newInputWire();
        // auto input3 = circ.newInputWire();
        // auto input4 = circ.newInputWire();
        // auto cmp = circ.addGate(common::utils::GateType::kMul3, input1, input2, input4);
        // circ.setAsOutput(cmp);
    // }

    // auto input = circ.newInputWire();
    // auto cmp = circ.addGate(common::utils::GateType::kLtz, input);
    // circ.setAsOutput(cmp);

    // for (int j = 0; j < nP; ++j) {
        // std::vector<wire_t> in1(10);
        // std::vector<wire_t> in2(10);
        // for (int i = 0; i < in1.size(); ++i) {
        //     in1[i] = circ.newInputWire();
        //     in2[i] = circ.newInputWire();
        // }
        // auto out = circ.addGate(common::utils::GateType::kDotprod, in1, in2);
        // auto out2 = circ.addGate(common::utils::GateType::kMul, out, in1[0]);
        // circ.setAsOutput(out);
    // }

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
    auto iter = opts["iter"].as<int>();
    auto latency = opts["latency"].as<double>();
    auto pid = opts["pid"].as<size_t>();
    auto threads = opts["threads"].as<size_t>();
    auto seed = opts["seed"].as<size_t>();
    auto repeat = opts["repeat"].as<size_t>();
    auto port = opts["port"].as<int>();

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
    StatsPoint init_start(*network);
    auto circ = generateCircuit(network, nP, pid, vec_size, iter).orderGatesByLevel();
    network->sync();
    StatsPoint init_end(*network);

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
    emp::PRG prg(&emp::zero_block, seed);
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed);
    auto preproc = off_eval.run(input_pid_map);
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    StatsPoint preproc_end(*network);

    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed);
    eval.setRandomInputs();
    for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
        eval.evaluateGatesAtDepth(i);
    }
    std::cout << "Online evaluation complete" << std::endl;
    network->sync();
    StatsPoint online_end(*network);

    StatsPoint end(*network);

    auto init_rbench = init_end - init_start;
    auto preproc_rbench = preproc_end - preproc_start;
    auto online_rbench = online_end - online_start;
    auto total_rbench = end - start;
    output_data["benchmarks"].push_back(init_rbench);
    output_data["benchmarks"].push_back(preproc_rbench);
    output_data["benchmarks"].push_back(online_rbench);
    output_data["benchmarks"].push_back(total_rbench);

    size_t init_bytes_sent = 0;
    for (const auto& val : init_rbench["communication"]) {
        init_bytes_sent += val.get<int64_t>();
    }
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

    // std::cout << "--- Repetition " << r + 1 << " ---" << std::endl;
    std::cout << "init time: " << init_rbench["time"] << " ms" << std::endl;
    std::cout << "init sent: " << init_bytes_sent << " bytes" << std::endl;
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
        ("vec-size,v", bpo::value<size_t>()->required(), "Number of gates at each level.")
        ("iter,i", bpo::value<int>()->default_value(1), "Number of iterations for message passing.")
        ("latency,l", bpo::value<double>()->required(), "Network latency in ms.")
        ("pid,p", bpo::value<size_t>()->required(), "Party ID.")
        ("threads,t", bpo::value<size_t>()->default_value(6), "Number of threads (recommended 6).")
        ("seed", bpo::value<size_t>()->default_value(200), "Value of the random seed.")
        ("net-config", bpo::value<std::string>(), "Path to JSON file containing network details of all parties.")
        ("localhost", bpo::bool_switch(), "All parties are on same machine.")
        ("port", bpo::value<int>()->default_value(10000), "Base port for networking.")
        ("output,o", bpo::value<std::string>(), "File to save benchmarks.")
        ("repeat,r", bpo::value<size_t>()->default_value(1), "Number of times to run benchmarks.");
  return desc;
}
// clang-format on

int main(int argc, char* argv[]) {
    auto prog_opts(programOptions());
    bpo::options_description cmdline("Benchmark online phase for multiplication gates.");
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