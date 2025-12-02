#pragma once

#include <emp-tool/emp-tool.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#include "../io/netmp.h"
#include "../utils/circuit.h"
#include "../utils/thread_pool.h"


#include "preproc.h"
#include "graphdb/rand_gen_pool.h"
#include "sharing.h"
#include "../utils/types.h"

using namespace common::utils;

namespace graphdb {

class OfflineEvaluator {
  int nP_;  
  int id_;
  int latency_;  // Network latency in microseconds
  bool use_pking_;  // Use king party for reconstruction
  RandGenPool rgen_;
  std::shared_ptr<io::NetIOMP> network_;
  common::utils::LevelOrderedCircuit circ_;
  std::shared_ptr<common::utils::ThreadPool> tpool_;
  PreprocCircuit preproc_;

  // Used for running common coin protocol. Returns common random PRG key which
  // is then used to generate randomness for common coin output.
  //emp::block commonCoinKey();
    

  public:
  OfflineEvaluator(int nP, int my_id, std::shared_ptr<io::NetIOMP> network,
                   common::utils::LevelOrderedCircuit circ, int threads, int seed = 200, int latency = 100, bool use_pking = true);

  // Generate sharing of a random unknown value.
  static void randomShare(int nP, int pid, RandGenPool& rgen, AuthAddShare& share,
                          std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec);

  // Generate sharing of a random value known to party. Should be called by
  // dealer when other parties call other variant.
  static void randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                AuthAddShare& share, Field secret,
                                std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec);

  // Set masks for each wire. Should be called before running any of the other
  // subprotocols.
  void setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map,
                         std::vector<Field>& rand_sh_sec);

  void setWireMasks(const std::unordered_map<common::utils::wire_t, int>& input_pid_map);

  PreprocCircuit getPreproc();
  
  // Efficiently runs above subprotocols.
  PreprocCircuit run(const std::unordered_map<common::utils::wire_t, int>& input_pid_map);
};
};  // namespace graphdb