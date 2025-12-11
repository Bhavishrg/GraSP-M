#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "../io/netmp.h"
#include "../utils/circuit.h"
#include "../utils/thread_pool.h"
#include "preproc.h"
#include "rand_gen_pool.h"
#include "sharing.h"
#include "../utils/types.h"

using namespace common::utils;

namespace graphdb {
class OnlineEvaluator {
  int nP_;
  int id_;
  int latency_;  // Network latency in microseconds
  bool use_pking_;  // Use king party for reconstruction
  RandGenPool rgen_;
  std::shared_ptr<io::NetIOMP> network_;
  PreprocCircuit preproc_;
  common::utils::LevelOrderedCircuit circ_;
  std::vector<AuthAddShare> wires_;
  AuthAddShare check;
  std::shared_ptr<common::utils::ThreadPool> tpool_;

  // Helper function to reconstruct shares towards a designated party
  void reconstructToParty(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                                const std::vector<AuthAddShare>& shares_list,
                                std::vector<Field>& reconstructed_list,
                                int target_party, int latency);

  // Helper function to reconstruct shares via king party or direct all-to-all
  void reconstruct(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                         std::vector<AuthAddShare>& shares_list,
                         std::vector<Field>& reconstructed_list, AuthAddShare& check,
                         bool via_pking, int latency,
                         std::vector<AuthAddShare>* tag_shares_list = nullptr,
                         std::vector<Field>* tag_reconstructed_list = nullptr);

  public:
  
  OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                  PreprocCircuit preproc,
                  common::utils::LevelOrderedCircuit circ,
                  int threads, int seed = 200, int latency = 100, bool use_pking = true);

  OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                  PreprocCircuit preproc,
                  common::utils::LevelOrderedCircuit circ,
                  std::shared_ptr<common::utils::ThreadPool> tpool, int seed = 200, int latency = 100, bool use_pking = true);

  void setInputs(const std::unordered_map<common::utils::wire_t, Field> &inputs);

  void setRandomInputs();
    
  void multEvaluate(const std::vector<common::utils::FIn2Gate> &mult_gates);

  void eqzEvaluate(const std::vector<common::utils::FIn1Gate> &eqz_gates);

  void recEvaluate(const std::vector<common::utils::FIn1Gate> &rec_gates);

  void permAndShEvaluate(const std::vector<common::utils::SIMDOGate> &permAndSh_gates);
  
  void cPermAndShEvaluate(const std::vector<common::utils::SIMDOGate> &cPermAndSh_gates);
  
  void amortzdPnSEvaluate(const std::vector<common::utils::SIMDOGate> &amortzdPnS_gates);
  
  void cAmortzdPnSEvaluate(const std::vector<common::utils::SIMDOGate> &cAmortzdPnS_gates);
  
  void rewireEvaluate(const std::vector<common::utils::SIMDOGate> &rewire_gates);

  void evaluateGatesAtDepth(size_t depth);

  std::vector<Field> getOutputs();

  // Evaluate online phase for circuit
  std::vector<Field> evaluateCircuit(const std::unordered_map<common::utils::wire_t, Field> &inputs);

  // Verify the check value for authentication
  void verify(AuthAddShare check_val);
  };

}; // namespace graphdb