#include <io/netmp.h>
#include <graphdb/offline_evaluator.h>
#include <graphdb/online_evaluator.h>
#include <utils/circuit.h>
#include <utils/thread_pool.h>

#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <omp.h>

#include "utils.h"
#include "graphutils.h"

using namespace graphdb;
using json = nlohmann::json;
namespace bpo = boost::program_options;
using common::utils::Ring;

void initializeGraph(std::shared_ptr<io::NetIOMP> network, int nP, int pid, 
                            size_t num_verts, size_t num_edges, int latency_usec, size_t seed,
                            std::shared_ptr<common::utils::ThreadPool> tpool) {

    std::cout << "Initializing Phase Starting" << std::endl;

    Ring nV = static_cast<Ring>(num_verts);
    Ring nE = static_cast<Ring>(num_edges);

    std::cout << "Generating scale-free graph with nV=" << nV << ", nE=" << nE << std::endl;
    auto edges = generate_scale_free(nV, nE, seed = 42);
    std::cout << "Generated " << edges.size() << " edges" << std::endl;
    
    std::cout << "Building daglist..." << std::endl;
    auto daglist = build_daglist(nV, edges);
    std::cout << "Built daglist with " << daglist.size() << " entries" << std::endl;

    std::cout << "Distributing daglist among " << nP << " parties..." << std::endl;
    auto dist_daglist = distribute_daglist(daglist, nP);
    std::cout << "Distribution complete" << std::endl;

    // Print information about this party's subgraph
    int party_idx = pid - 1;  // Convert 1-indexed pid to 0-indexed
    std::cout << "\n=== Party " << pid << " Subgraph Info ===" << std::endl;
    std::cout << "Subgraph size: " << dist_daglist.SubgraphSizes[party_idx] << std::endl;
    std::cout << "Vertex entries: " << dist_daglist.SubgVertexLists[party_idx].size() << std::endl;
    std::cout << "Edge entries: " << dist_daglist.SubgEdgeLists[party_idx].size() << std::endl;
    std::cout << "sigg size: " << dist_daglist.sigg[party_idx].size() << std::endl;
    std::cout << "sigs size: " << dist_daglist.sigs[party_idx].size() << std::endl;
    std::cout << "sigd size: " << dist_daglist.sigd[party_idx].size() << std::endl;
    std::cout << "sigv size: " << dist_daglist.sigv[party_idx].size() << std::endl;
    std::cout << "================================\n" << std::endl;

    // INITIALIZATION PHASE: Every party sends their position maps to every other party
    if (pid != 0) {
        // Prepare data to send: sigg, sigs, sigd, sigv concatenated
        size_t sigg_size = dist_daglist.sigg[party_idx].size();
        size_t sigs_size = dist_daglist.sigs[party_idx].size();
        size_t sigd_size = dist_daglist.sigd[party_idx].size();
        size_t sigv_size = dist_daglist.sigv[party_idx].size();
        size_t total_size = sigg_size + sigs_size + sigd_size + sigv_size;
        
        std::vector<Ring> perm_send;
        perm_send.reserve(total_size);
        
        // Concatenate all position maps
        perm_send.insert(perm_send.end(), dist_daglist.sigg[party_idx].begin(), dist_daglist.sigg[party_idx].end());
        perm_send.insert(perm_send.end(), dist_daglist.sigs[party_idx].begin(), dist_daglist.sigs[party_idx].end());
        perm_send.insert(perm_send.end(), dist_daglist.sigd[party_idx].begin(), dist_daglist.sigd[party_idx].end());
        perm_send.insert(perm_send.end(), dist_daglist.sigv[party_idx].begin(), dist_daglist.sigv[party_idx].end());

        std::cout << "Sending position maps (total size: " << total_size << " elements, " 
                  << total_size * sizeof(Ring) << " bytes)" << std::endl;

        // Step 1: Exchange sizes first
        std::vector<size_t> all_sizes(nP);
        all_sizes[party_idx] = total_size;

        // Use thread pool to send and receive sizes in parallel
        auto size_send_future = tpool->enqueue([&]() {
            for (int i = 1; i <= nP; ++i) {
                if (i != pid) {
                    network->send(i, &total_size, sizeof(size_t));
                    network->flush(i);
                }
            }
        });

        auto size_recv_future = tpool->enqueue([&]() {
            usleep(latency_usec);
            for (int i = 1; i <= nP; ++i) {
                if (i != pid) {
                    network->recv(i, &all_sizes[i - 1], sizeof(size_t));
                }
            }
        });

        // Wait for size exchange to complete
        if (size_send_future.valid()) {
            size_send_future.wait();
        }
        if (size_recv_future.valid()) {
            size_recv_future.wait();
        }

        // Step 2: Allocate receive buffers based on actual sizes
        std::vector<std::vector<Ring>> perm_recv(nP);
        for (int i = 0; i < nP; ++i) {
            if (i != party_idx) {
                perm_recv[i].resize(all_sizes[i]);
            }
        }

        // Copy own data first
        perm_recv[party_idx] = perm_send;

        // Step 3: Exchange position maps with all other parties
        // Use thread pool to send and receive position maps in parallel
        auto send_future = tpool->enqueue([&]() {
            for (int i = 1; i <= nP; ++i) {
                if (i != pid) {
                    network->send(i, perm_send.data(), perm_send.size() * sizeof(Ring));
                    network->flush(i);
                }
            }
        });

        auto recv_future = tpool->enqueue([&]() {
            usleep(latency_usec);
            for (int i = 1; i <= nP; ++i) {
                if (i != pid) {
                    network->recv(i, perm_recv[i - 1].data(), perm_recv[i - 1].size() * sizeof(Ring));
                }
            }
        });

        // Wait for both send and receive operations to complete
        if (send_future.valid()) {
            send_future.wait();
        }
        if (recv_future.valid()) {
            recv_future.wait();
        }

        std::cout << "Position maps exchanged with all parties" << std::endl;
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
    auto num_verts = opts["num-verts"].as<size_t>();
    auto num_edges = opts["num-edges"].as<size_t>();
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

    // Create thread pool for parallelization
    auto tpool = std::make_shared<common::utils::ThreadPool>(threads);

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
                              {"num_verts", num_verts},
                              {"num_edges", num_edges},
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
    int latency_usec = static_cast<int>(latency * 1000);  // Convert latency from ms to microseconds
    initializeGraph(network, nP, pid, num_verts, num_edges, latency_usec, seed, tpool);
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
        ("num-verts", bpo::value<size_t>()->required(), "Number of vertices in the graph.")
        ("num-edges", bpo::value<size_t>()->required(), "Number of edges in the graph.")
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
    bpo::options_description cmdline("Benchmark initialization phase with scale-free graph");
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

// usage: ./../run.sh initialization_grasp --num-parties 2 --num-verts 10000 --num-edges 100000