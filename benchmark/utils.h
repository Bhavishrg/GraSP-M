#pragma once

#include <io/netmp.h>
#include <utils/circuit.h>

#include <array>
#include <chrono>
#include <nlohmann/json.hpp>
#include <string>

struct TimePoint {
  using timepoint_t = std::chrono::high_resolution_clock::time_point;
  using timeunit_t = std::chrono::duration<double, std::milli>;

  TimePoint();
  double operator-(const TimePoint& rhs) const;

  timepoint_t time;
};

struct CommPoint {
  std::vector<uint64_t> stats;

  explicit CommPoint(io::NetIOMP& network);
  std::vector<uint64_t> operator-(const CommPoint& rhs) const;
};

class StatsPoint {
  TimePoint tpoint_;
  CommPoint cpoint_;

 public:
  explicit StatsPoint(io::NetIOMP& network);
  nlohmann::json operator-(const StatsPoint& rhs);
};

bool saveJson(const nlohmann::json& data, const std::string& fpath);
int64_t peakVirtualMemory();
int64_t peakResidentSetSize();
void initNTL(size_t num_threads);
void increaseSocketBuffers(io::NetIOMP* network, int buffer_size);
std::shared_ptr<io::NetIOMP> createNetwork(int pid, int nP, int latency, int port, 
                                            bool localhost, const std::string& net_config_path);

// Sub-circuit building functions
std::vector<common::utils::wire_t> addSubCircPropagate(
    common::utils::Circuit<common::utils::Ring>& circ,
    const std::vector<common::utils::wire_t>& position_map_shares,
    const std::vector<common::utils::wire_t>& data_values,
    size_t num_groups,
    std::vector<std::vector<int>> permutation, bool in = false);

std::vector<common::utils::wire_t> addSubCircGather(
    common::utils::Circuit<common::utils::Ring>& circ,
    const std::vector<common::utils::wire_t>& position_map_shares,
    const std::vector<common::utils::wire_t>& data_values,
    size_t num_groups,
    std::vector<std::vector<int>> permutation);

std::vector<std::vector<common::utils::wire_t>> addSubCircPermList(
    common::utils::Circuit<common::utils::Ring>& circ,
    const std::vector<common::utils::wire_t>& position_map_shares,
    const std::vector<std::vector<common::utils::wire_t>>& payloads,
    std::vector<std::vector<int>> permutation);
