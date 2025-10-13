#pragma once

#include <emp-tool/emp-tool.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#include "../io/netmp.h"
#include "../utils/circuit.h"


#include "preproc.h"
#include "graphdb/rand_gen_pool.h"
#include "sharing.h"
#include "../utils/types.h"

using namespace common::utils;

namespace graphdb {
class OfflineEvaluator {
  int nP_;  
  int id_;
  RandGenPool rgen_;
  std::shared_ptr<io::NetIOMP> network_;
  common::utils::LevelOrderedCircuit circ_;
  std::shared_ptr<ThreadPool> tpool_;
  PreprocCircuit<Ring> preproc_;

  // Used for running common coin protocol. Returns common random PRG key which
  // is then used to generate randomness for common coin output.
  //emp::block commonCoinKey();

  public:
  OfflineEvaluator(int nP, int my_id, std::shared_ptr<io::NetIOMP> network,
                   common::utils::LevelOrderedCircuit circ, int threads, int seed = 200);

  // Generate sharing of a random unknown value.
  static void randomShare(int nP, int pid, RandGenPool& rgen, AddShare<Ring>& share, TPShare<Ring>& tpShare);

  // Generate sharing of a random value known to party. Should be called by
  // dealer when other parties call other variant.
  static void randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                AddShare<Ring>& share, TPShare<Ring>& tpShare, Ring secret,
                                std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec);

  // Generate sharing of a random unknown permutation.
  static void randomPermutation(int nP, int pid, RandGenPool& rgen, std::vector<int>& pi, size_t& vec_size);

  void generateShuffleDeltaVector(int nP, int pid, RandGenPool& rgen, std::vector<AddShare<Ring>>& delta,
                                  std::vector<TPShare<Ring>>& tp_a, std::vector<TPShare<Ring>>& tp_b,
                                  std::vector<TPShare<Ring>>& tp_c, std::vector<std::vector<int>>& tp_pi_all,
                                  size_t& vec_size, std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec);


  void generatePermAndShDeltaVector(int nP, int pid, RandGenPool& rgen, int owner, std::vector<AddShare<Ring>>& delta,
                                    std::vector<TPShare<Ring>>& tp_a, std::vector<TPShare<Ring>>& tp_b,
                                    std::vector<int>& pi, size_t& vec_size, std::vector<Ring>& delta_sh, size_t& idx_delta_sh);

  // Following methods implement various preprocessing subprotocols.

  // Set masks for each wire. Should be called before running any of the other
  // subprotocols.
  void setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map,
                         std::vector<Ring>& rand_sh_sec,
                         std::vector<std::vector<Ring>>& delta_sh);

  void setWireMasks(const std::unordered_map<common::utils::wire_t, int>& input_pid_map);

  PreprocCircuit<Ring> getPreproc();

  // Efficiently runs above subprotocols.
  PreprocCircuit<Ring> run(const std::unordered_map<common::utils::wire_t, int>& input_pid_map);
};

};  // namespace graphdb
