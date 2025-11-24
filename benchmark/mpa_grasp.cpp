#include <io/netmp.h>
#include <grasp/offline_evaluator.h>
#include <grasp/online_evaluator.h>
#include <utils/circuit.h>

#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <omp.h>

#include "utils.h"

#include <iomanip>

using namespace grasp;
using json = nlohmann::json;
namespace bpo = boost::program_options;

common::utils::Circuit<Ring> generateCircuit(std::shared_ptr<io::NetIOMP> &network, int nP, int pid, size_t vec_size, int iter) {

    std::cout << "Generating circuit" << std::endl;
    
    common::utils::Circuit<Ring> circ;

    size_t num_vert = 0.1 * vec_size;
    size_t num_edge = vec_size - num_vert;
    std::vector<size_t> subg_num_vert(nP);
    std::vector<size_t> subg_num_edge(nP);
    for (int i = 0; i < subg_num_vert.size(); ++i) {
        if (i != nP - 1) {
            subg_num_vert[i] = num_vert / nP;
            subg_num_edge[i] = num_edge / nP;
        } else {
            subg_num_vert[i] = num_vert / nP + num_vert % nP;
            subg_num_edge[i] = num_edge / nP + num_edge % nP;
        }
    }

    std::cout << "num_vert " << num_vert << " num_edge " << num_edge << std::endl;

    // INPUT SHARING PHASE
    std::vector<wire_t> full_vertex_list(num_vert);
    for (int i = 0; i < num_vert; ++i) {
        full_vertex_list[i] = circ.newInputWire();
    }
    std::vector<std::vector<wire_t>> subg_edge_list(nP);
    for (int i = 0; i < subg_edge_list.size(); ++i) {
        std::vector<wire_t> subg_edge_list_party(subg_num_edge[i]);
        for (int j = 0; j < subg_edge_list_party.size(); ++j) {
            subg_edge_list_party[j] = circ.newInputWire();
        }
        subg_edge_list[i] = subg_edge_list_party;
    }

    // MESSAGE PASSING

    std::vector<std::vector<int>> permutation;
    std::vector<int> tmp_perm(num_vert);
    for (int i = 0; i < num_vert; ++i) {
        tmp_perm[i] = i;
    }
    permutation.push_back(tmp_perm);
    if (pid == 0) {
        for (int i = 1; i < nP; ++i) {
            permutation.push_back(tmp_perm);
        }
    }

    // DECOMPOSE
    std::vector<wire_t> subg_sorted_vert_list(num_vert, 0);
    for (int i = 0; i < nP; ++i) {
        // SUB GRAPH GEN
        auto num_subg_vert = std::min(num_vert, 2 * subg_num_edge[i]);
        std::vector<wire_t> subg_dag_list_party;
        subg_dag_list_party.reserve(num_subg_vert + subg_edge_list[i].size());
        std::vector<wire_t> subg_permuted_vert_list(subg_sorted_vert_list.size());
        for (int j = 0; j < subg_permuted_vert_list.size(); ++j) {
            subg_permuted_vert_list[j] = full_vertex_list[j];
        }
        subg_dag_list_party.insert(subg_dag_list_party.end(), subg_permuted_vert_list.begin(), subg_permuted_vert_list.begin() + num_subg_vert);
        subg_dag_list_party.insert(subg_dag_list_party.end(), subg_edge_list[i].begin(), subg_edge_list[i].end());

        std::vector<std::vector<int>> subg_permutation;
        std::vector<int> subg_tmp_perm(subg_dag_list_party.size());
        for (int i = 0; i < subg_tmp_perm.size(); ++i) {
            subg_tmp_perm[i] = i;
        }
        subg_permutation.push_back(subg_tmp_perm);
        if (pid == 0) {
            for (int i = 1; i < nP; ++i) {
                subg_permutation.push_back(subg_tmp_perm);
            }
        }

        // PROPAGATE
        std::vector<wire_t> tmp(subg_dag_list_party.size());
        for (int j = num_subg_vert - 1; j > 0; --j) {
            tmp[j] = circ.addGate(common::utils::GateType::kSub, subg_dag_list_party[j], subg_dag_list_party[j - 1]);
        }
        auto permAndSh_list1 = circ.addMGate(common::utils::GateType::kPermAndSh, tmp, subg_permutation, i + 1);
        std::vector<wire_t> propagate_list(permAndSh_list1.size());
        for (int j = 0; j < permAndSh_list1.size(); ++j) {
            propagate_list[j] = permAndSh_list1[j];
        }
        std::vector<wire_t> tmp1(propagate_list.size());
        for (int j = 1; j < permAndSh_list1.size(); ++j) {
            tmp1[j] = circ.addGate(common::utils::GateType::kAdd, propagate_list[j - 1], propagate_list[j]);
        }
        for (int j = propagate_list.size() - 1; j > 0; --j) {
            propagate_list[j] = circ.addGate(common::utils::GateType::kSub, tmp1[j], propagate_list[j]);
        }

        // SRC TO DST
        auto permAndSh_list2 = circ.addMGate(common::utils::GateType::kPermAndSh, propagate_list, subg_permutation, i + 1);
        std::vector<wire_t> tmp2(permAndSh_list2.size());
        for (int j = 1; j < permAndSh_list2.size(); ++j) {
            tmp2[j] = permAndSh_list2[j];
        }

        // GATHER
        std::vector<wire_t> permAndSh_list3(tmp2.size());
        for (int j = 1; j < permAndSh_list3.size(); ++j) {
            permAndSh_list3[j] = circ.addGate(common::utils::GateType::kAdd, tmp2[j], tmp2[j - 1]);
        }
        auto permAndSh_list4 = circ.addMGate(common::utils::GateType::kPermAndSh, permAndSh_list3, subg_permutation, i + 1);
        std::vector<wire_t> tmp3(permAndSh_list4.size());
        for (int j = 0; j < permAndSh_list4.size(); ++j) {
            tmp3[j] = permAndSh_list4[j];
        }
        std::vector<wire_t> gather_list(tmp3.size());
        for (int j = gather_list.size() - 1; j > 0; --j) {
            gather_list[j] = circ.addGate(common::utils::GateType::kSub, tmp3[j], tmp3[j - 1]);
        }

        // APPLYV - PAGERANK
        for (int j = 0; j < gather_list.size(); ++j) {
            auto pgr = circ.addConstOpGate(common::utils::GateType::kConstMul, gather_list[j], Ring(1));
            gather_list[j] = circ.addConstOpGate(common::utils::GateType::kConstAdd, pgr, Ring(1));
            circ.setAsOutput(gather_list[j]);
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
    auto iter = opts["iter"].as<int>();
    auto latency = opts["latency"].as<double>();
    auto pid = opts["pid"].as<size_t>();
    auto threads = opts["threads"].as<size_t>();
    auto seed = opts["seed"].as<size_t>();
    auto repeat = opts["repeat"].as<size_t>();
    auto port = opts["port"].as<int>();

    omp_set_nested(1);
    // omp_set_num_threads(nP);
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
    int latency_ms = static_cast<int>(latency);  // Convert latency from double to int milliseconds
    OfflineEvaluator off_eval(nP, pid, network, circ, threads, seed, latency_ms);
    auto preproc = off_eval.run(input_pid_map);
    std::cout << "Preprocessing complete" << std::endl;
    network->sync();
    StatsPoint preproc_end(*network);

    std::cout << "Starting online evaluation" << std::endl;
    StatsPoint online_start(*network);
    OnlineEvaluator eval(nP, pid, network, std::move(preproc), circ, threads, seed, latency_ms);
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
    std::cout << std::fixed << std::setprecision(2);
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
        ("latency,l", bpo::value<double>()->default_value(100.0), "Network latency in ms.")
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