#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "../io/netmp.h"
#include "../utils/circuit.h"
#include "preproc.h"
#include "rand_gen_pool.h"
#include "sharing.h"
#include "../utils/types.h"

using namespace common::utils;

namespace graphdb {
  class OnlineEvaluator {
    int nP_;
    int id_;
    RandGenPool rgen_;
    std::shared_ptr<io::NetIOMP> network_;
    PreprocCircuit<Ring> preproc_;
    common::utils::LevelOrderedCircuit circ_;
    std::vector<Ring> wires_;
    std::shared_ptr<ThreadPool> tpool_;

    // write reconstruction function
  public:
    OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                    PreprocCircuit<Ring> preproc,
                    common::utils::LevelOrderedCircuit circ,
                    int threads, int seed = 200);

    OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                    PreprocCircuit<Ring> preproc,
                    common::utils::LevelOrderedCircuit circ,
                    std::shared_ptr<ThreadPool> tpool, int seed = 200);

    void setInputs(const std::unordered_map<common::utils::wire_t, Ring> &inputs);

    void setRandomInputs();

    void evaluateGatesAtDepthPartySend(size_t depth, std::vector<Ring> &mult_vals);

    void evaluateGatesAtDepthPartyRecv(size_t depth, std::vector<Ring> &mult_vals);

    void evaluateGatesAtDepth(size_t depth);

    void shuffleEvaluate(const std::vector<common::utils::SIMDOGate> &shuffle_gates);

    void permAndShEvaluate(const std::vector<common::utils::SIMDOGate> &permAndSh_gates);

    std::vector<Ring> getOutputs();

    Ring reconstruct(AddShare<Ring> &shares);

    // Evaluate online phase for circuit
    std::vector<Ring> evaluateCircuit(const std::unordered_map<common::utils::wire_t, Ring> &inputs);
  };

}; // namespace graphdb
