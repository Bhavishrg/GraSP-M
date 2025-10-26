#include "utils.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pE.h>
#include <sys/socket.h>
#include <cerrno>

#include <fstream>
#include <iostream>

void increaseSocketBuffers(io::NetIOMP* network, int buffer_size) {
    int actual_sndbuf = 0, actual_rcvbuf = 0;
    socklen_t optlen = sizeof(int);
    bool first_socket = true;
    
    for (int i = 0; i < network->nP; ++i) {
        if (i != network->party) {
            // Get the send and receive channels
            auto send_channel = network->getSendChannel(i);
            auto recv_channel = network->getRecvChannel(i);
            
            if (send_channel) {
                int fd = send_channel->consocket;
                if (fd >= 0) {
                    int ret1 = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
                    int ret2 = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
                    
                    if (first_socket) {
                        // Verify actual buffer sizes set
                        getsockopt(fd, SOL_SOCKET, SO_SNDBUF, &actual_sndbuf, &optlen);
                        getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &actual_rcvbuf, &optlen);
                        first_socket = false;
                        
                        if (ret1 != 0 || ret2 != 0) {
                            std::cout << "Warning: setsockopt failed (errno: " << errno << ")" << std::endl;
                        }
                    }
                }
            }
            
            if (recv_channel && recv_channel != send_channel) {
                int fd = recv_channel->consocket;
                if (fd >= 0) {
                    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
                    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
                }
            }
        }
    }
    std::cout << "Requested socket buffers: " << buffer_size << " bytes" << std::endl;
    if (actual_sndbuf > 0 || actual_rcvbuf > 0) {
        std::cout << "Actual socket buffers: SNDBUF=" << actual_sndbuf 
                  << " bytes, RCVBUF=" << actual_rcvbuf << " bytes" << std::endl;
        if (actual_sndbuf < buffer_size || actual_rcvbuf < buffer_size) {
            std::cout << "Warning: Socket buffers were capped by system limits!" << std::endl;
            std::cout << "Consider running Docker with: --sysctl net.core.rmem_max=" << buffer_size 
                      << " --sysctl net.core.wmem_max=" << buffer_size << std::endl;
        }
    }
}

TimePoint::TimePoint() : time(timepoint_t::clock::now()) {}

double TimePoint::operator-(const TimePoint& rhs) const {
  return std::chrono::duration_cast<timeunit_t>(time - rhs.time).count();
}

CommPoint::CommPoint(io::NetIOMP& network) : stats(network.nP) {
  for (size_t i = 0; i < network.nP; ++i) {
    if (i != network.party) {
      stats[i] = network.get(i, false)->counter + network.get(i, true)->counter;
    }
  }
}

std::vector<uint64_t> CommPoint::operator-(const CommPoint& rhs) const {
  std::vector<uint64_t> res(stats.size());
  for (size_t i = 0; i < stats.size(); ++i) {
    res[i] = stats[i] - rhs.stats[i];
  }
  return res;
}

StatsPoint::StatsPoint(io::NetIOMP& network) : cpoint_(network) {}

nlohmann::json StatsPoint::operator-(const StatsPoint& rhs) {
  return {{"time", tpoint_ - rhs.tpoint_},
          {"communication", cpoint_ - rhs.cpoint_}};
}

bool saveJson(const nlohmann::json& data, const std::string& fpath) {
  std::ofstream fout;
  //fout.open(fpath, std::fstream::out);
  fout.open(fpath, std::fstream::app);
  if (!fout.is_open()) {
    std::cerr << "Could not open save file at " << fpath << std::endl;
    return false;
  }

  fout << data;
  fout << std::endl;
  fout.close();

  std::cout << "Saved data in " << fpath << std::endl;

  return true;
}

void initNTL(size_t num_threads) {
  NTL::ZZ_p::init(NTL::conv<NTL::ZZ>("18446744073709551616"));
  NTL::ZZ_pX P(NTL::INIT_MONO, 47);
  NTL::SetCoeff(P, 5);
  NTL::SetCoeff(P, 0);
  NTL::ZZ_pE::init(P);

  NTL::SetNumThreads(num_threads);
}

#ifdef __APPLE__

#include <mach/mach.h>

int64_t peakResidentSetSize() {
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

  kern_return_t ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                reinterpret_cast<task_info_t>(&info), &count);
  if (ret != KERN_SUCCESS || count != MACH_TASK_BASIC_INFO_COUNT) {
    return -1;
  }

  return info.resident_size_max;
}

int64_t peakVirtualMemory() {
  // No way to get peak virtual memory usage on OSX.
  return peakResidentSetSize();
}
#elif __linux__
// Reference: https://gist.github.com/k3vur/4169316
int64_t getProcStatus(const std::string& key) {
  int64_t value = 0;

  const char* filename = "/proc/self/status";

  std::ifstream procfile(filename);
  std::string word;
  while (procfile.good()) {
    procfile >> word;
    if (word == key) {
      procfile >> value;
      break;
    }

    // Skip to end of line.
    procfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  if (procfile.fail()) {
    return -1;
  }

  return value;
}

int64_t peakVirtualMemory() { return getProcStatus("VmPeak:"); }

int64_t peakResidentSetSize() { return getProcStatus("VmHWM:"); }
#else
int64_t peakVirtualMemory() { return -1; }

int64_t peakResidentSetSize() { return -1; }
#endif

// Sub-circuit building functions

std::vector<common::utils::wire_t> addSubCircPropagate(
    common::utils::Circuit<common::utils::Ring>& circ,
    const std::vector<common::utils::wire_t>& position_map_shares,
    const std::vector<common::utils::wire_t>& data_values,
    size_t num_groups,
    std::vector<std::vector<int>> permutation) {
    
    size_t vec_size = position_map_shares.size();
    
    // Validate num_groups < vec_size
    if (num_groups >= vec_size) {
        throw std::invalid_argument("num_groups must be less than vec_size");
    }
    
    // Step 1: Compute differences for group boundaries
    // data_values'[0] = data_values[0]
    // data_values'[i] = data_values[i] - data_values[i-1] for i = 1 to num_groups-1
    // data_values'[i] = 0 for i >= num_groups
    std::vector<common::utils::wire_t> data_values_diff(vec_size);
    
    for (size_t i = 0; i < vec_size; ++i) {
        if (i == 0) {
            // data_values'[0] = data_values[0]
            data_values_diff[i] = data_values[i];
        } else if (i < num_groups) {
            // For i from 1 to num_groups-1: compute differences
            // data_values'[i] = data_values[i] - data_values[i-1]
            data_values_diff[i] = circ.addGate(common::utils::GateType::kSub, data_values[i], data_values[i - 1]);
        } else {
            // For i >= num_groups: set to 0
            // data_values'[i] = 0
            data_values_diff[i] = 0;
        }
    }

    // Step 2: Shuffle both position_map and data_values'

    auto shuffled_position_map = circ.addMGate(common::utils::GateType::kShuffle, position_map_shares, permutation);
    auto shuffled_data_values = circ.addMGate(common::utils::GateType::kShuffle, data_values_diff, permutation);

    // Step 3: Reconstruct position map
    std::vector<common::utils::wire_t> position_map_reconstructed(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        position_map_reconstructed[i] = circ.addGate(common::utils::GateType::kRec, shuffled_position_map[i]);
    }

    // Step 4: Rewire shuffled data_values using reconstructed position map
    std::vector<std::vector<common::utils::wire_t>> payloads = {shuffled_data_values};
    auto rewired_outputs = circ.addRewireGate(position_map_reconstructed, payloads);
    auto reordered_data = rewired_outputs[0];

    // Step 5: Compute prefix sum of reordered data
    std::vector<common::utils::wire_t> prefix_sum(vec_size);
    prefix_sum[0] = reordered_data[0];
    for (size_t i = 1; i < vec_size; ++i) {
        prefix_sum[i] = circ.addGate(common::utils::GateType::kAdd, prefix_sum[i - 1], reordered_data[i]);
    }

    return prefix_sum;
}

std::vector<common::utils::wire_t> addSubCircGather(
    common::utils::Circuit<common::utils::Ring>& circ,
    const std::vector<common::utils::wire_t>& position_map_shares,
    const std::vector<common::utils::wire_t>& data_values,
    size_t num_groups,
    std::vector<std::vector<int>> permutation) {
    
    size_t vec_size = position_map_shares.size();
    
    // Validate num_groups < vec_size
    if (num_groups >= vec_size) {
        throw std::invalid_argument("num_groups must be less than vec_size");
    }

    // Step 1: Compute prefix sum of data values
    std::vector<common::utils::wire_t> prefix_sum(vec_size);
    prefix_sum[0] = data_values[0];
    for (size_t i = 1; i < vec_size; ++i) {
        prefix_sum[i] = circ.addGate(common::utils::GateType::kAdd, prefix_sum[i - 1], data_values[i]);
    }

    // Step 2: Shuffle both position_map and prefix_sum

    auto shuffled_position_map = circ.addMGate(common::utils::GateType::kShuffle, position_map_shares, permutation);
    auto shuffled_prefix_sum = circ.addMGate(common::utils::GateType::kShuffle, prefix_sum, permutation);

    // Step 3: Reconstruct position map
    std::vector<common::utils::wire_t> position_map_reconstructed(vec_size);
    for (size_t i = 0; i < vec_size; ++i) {
        position_map_reconstructed[i] = circ.addGate(common::utils::GateType::kRec, shuffled_position_map[i]);
    }

    // Step 4: Rewire shuffled prefix_sum using reconstructed position map
    std::vector<std::vector<common::utils::wire_t>> payloads = {shuffled_prefix_sum, data_values};
    auto rewired_outputs = circ.addRewireGate(position_map_reconstructed, payloads);
    auto reordered_data = rewired_outputs[0];
    auto rewired_data_values = rewired_outputs[1];

    // Step 5: Compute differences
    // data_values'[0] = reordered_data[0] - data_values[0]
    // data_values'[i] = reordered_data[i] - reordered_data[i-1] - data_values[i] for i = 1 to num_groups-1
    // data_values'[i] = 0 for i >= num_groups
    std::vector<common::utils::wire_t> data_values_diff(vec_size);
    
    for (size_t i = 0; i < vec_size; ++i) {
        if (i == 0) {
            // data_values'[0] = reordered_data[0] - data_values[0]
            data_values_diff[i] = circ.addGate(common::utils::GateType::kSub, reordered_data[i], rewired_data_values[i]);
        } else if (i < num_groups) {
            // For i from 1 to num_groups-1: compute differences
            // data_values'[i] = reordered_data[i] - reordered_data[i-1] - data_values[i]
            data_values_diff[i] = circ.addGate(common::utils::GateType::kSub, reordered_data[i], reordered_data[i - 1]);
            data_values_diff[i] = circ.addGate(common::utils::GateType::kSub, data_values_diff[i], rewired_data_values[i]);
        } else {
            // For i >= num_groups: set to 0
            // data_values'[i] = 0
            data_values_diff[i] = 0;
        }
    }

    return data_values_diff;
}
