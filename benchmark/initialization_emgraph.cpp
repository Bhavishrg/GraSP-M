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

void initializePermutations(std::shared_ptr<io::NetIOMP> network, int nP, int pid, size_t vec_size) {

    std::cout << "Initializing Phase Starting" << std::endl;

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

    // INITIALIZATION PHASE
    if (pid != 0) {
        auto subg_dag_list_size = std::min(num_vert, 2 * subg_num_edge[pid - 1]) + subg_num_edge[pid - 1];
        std::vector<int> perm_send(num_vert + 3 * subg_dag_list_size);
        std::vector<std::vector<int>> perm_recv(nP, std::vector<int>(perm_send.size()));

        std::vector<int> rand_perm_g(num_vert);
        std::vector<int> perm_g(num_vert);
        for (int i = 0; i < perm_g.size(); ++i) {
            perm_g[i] = i;
            rand_perm_g[i] = i;
        }
        for (int i = 0; i < perm_g.size(); ++i) {
            perm_send[i] = perm_g[rand_perm_g[i]];
        }

        std::vector<int> rand_perm_s(subg_dag_list_size);
        std::vector<int> perm_s(subg_dag_list_size);
        for (int i = 0; i < perm_s.size(); ++i) {
            perm_s[i] = i;
            rand_perm_s[i] = i;
        }
        for (int i = 0; i < perm_s.size(); ++i) {
            perm_send[i + perm_g.size()] = perm_s[rand_perm_s[i]];
        }

        std::vector<int> rand_perm_d(subg_dag_list_size);
        std::vector<int> perm_d(subg_dag_list_size);
        for (int i = 0; i < perm_d.size(); ++i) {
            perm_d[i] = i;
            rand_perm_d[i] = i;
        }
        for (int i = 0; i < perm_d.size(); ++i) {
            perm_send[i + perm_g.size() + perm_s.size()] = perm_d[rand_perm_d[i]];
        }

        std::vector<int> rand_perm_v(subg_dag_list_size);
        std::vector<int> perm_v(subg_dag_list_size);
        for (int i = 0; i < perm_v.size(); ++i) {
            perm_v[i] = i;
            rand_perm_v[i] = i;
        }
        for (int i = 0; i < perm_v.size(); ++i) {
            perm_send[i + perm_g.size() + perm_s.size() + perm_d.size()] = perm_v[rand_perm_v[i]];
        }

        std::cout << "Sending permutations" << std::endl;

        #pragma omp parallel for
        for (int i = 1; i <= nP; ++i) {
            if (i != pid) {
                omp_set_num_threads(2);
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        network->send(i, perm_send.data(), perm_send.size() * sizeof(int));
                        network->flush(i);
                    }
                    #pragma omp section
                    {
                        usleep(250);
                        network->recv(i, perm_recv[i - 1].data(), perm_recv[i - 1].size() * sizeof(int));
                    }
                }
            } else {
                std::copy(perm_send.begin(), perm_send.end(), perm_recv[i - 1].begin());
            }
        }
    }

    std::cout << "Initialization done" << std::endl;
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
    initializePermutations(network, nP, pid, vec_size);
    network->sync();
    StatsPoint init_end(*network);

    StatsPoint end(*network);

    auto init_rbench = init_end - init_start;
    auto total_rbench = end - start;
    output_data["benchmarks"].push_back(init_rbench);
    output_data["benchmarks"].push_back(total_rbench);

    size_t init_bytes_sent = 0;
    for (const auto& val : init_rbench["communication"]) {
        init_bytes_sent += val.get<int64_t>();
    }
    size_t total_bytes_sent = 0;
    for (const auto& val : total_rbench["communication"]) {
        total_bytes_sent += val.get<int64_t>();
    }

    // std::cout << "--- Repetition " << r + 1 << " ---" << std::endl;
    std::cout << "init time: " << init_rbench["time"] << " ms" << std::endl;
    std::cout << "init sent: " << init_bytes_sent << " bytes" << std::endl;
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