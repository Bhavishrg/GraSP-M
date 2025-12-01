#include "offline_evaluator.h"

#include <NTL/BasicThreadPool.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>

#include "../utils/helpers.h"

namespace graphdb {
OfflineEvaluator::OfflineEvaluator(int nP, int my_id,
                                   std::shared_ptr<io::NetIOMP> network,
                                   common::utils::LevelOrderedCircuit circ,
                                   int threads, int seed, int latency, bool use_pking)
    : nP_(nP),
      id_(my_id),
      latency_(latency),
      use_pking_(use_pking),
      rgen_(my_id, seed), 
      network_(std::move(network)),
      circ_(std::move(circ))
      // preproc_(circ.num_gates)

      { } // tpool_ = std::make_shared<ThreadPool>(threads); }


void OfflineEvaluator::randomShare(int nP, int pid, RandGenPool& rgen, AuthAddShare<Ring>& share,
                                         std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  Ring val = Ring(0);
  Ring valn = Ring(0);
  Ring tag = Ring(0);
  Ring tagn = Ring(0);
  if (pid == 0) {
    share.pushValue(Ring(0));
    share.pushTag(Ring(0));
/*     tpShare.pushValues(Ring(0));
    for (int i = 1; i <= nP; i++) {
      rgen.pi(i).random_data(&val, sizeof(Ring));
      tpShare.pushValues(val);
    }  */
  } else {
    rgen.p0().random_data(&val, sizeof(Ring));
    share.pushValue(val);
    valn += val;
    if(pid != nP) {
      rgen.p0().random_data(&tag, sizeof(Ring));
      share.pushTag(tag);
      tagn += tag;
    }
    else {
      tag = global_key*valn - tagn;
      share.pushTag(tag);
      rand_sh_sec.push_back(tag);
      idx_rand_sh_sec++;
    }
  }
}

void OfflineEvaluator::randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                         AuthAddShare<Ring>& share, Ring secret,
                                         std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  Ring val = Ring(0);
  Ring valn = Ring(0);
  Ring tag = Ring(0);
  Ring tagn = Ring(0);
                                          
  if (pid != nP) {
    rgen.p0().random_data(&val, sizeof(Ring));
    valn += val;
    share.pushValue(val);

    rgen.p0().random_data(&tag, sizeof(Ring));
    share.pushTag(tag);
    tagn += tag;
  } else {
    valn = secret - valn;
    rand_sh_sec.push_back(valn);
    share.pushValue(rand_sh_sec[idx_rand_sh_sec]);
    idx_rand_sh_sec++;

    tag = global_key*secret - tagn; //global_key use has to be fixed.
    share.pushTag(tag);
    rand_sh_sec.push_back(tag);
    idx_rand_sh_sec++;
  }
}

void OfflineEvaluator::setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map, 
                                         std::vector<Ring>& rand_sh_sec) {
  size_t idx_rand_sh_sec = 0;
  size_t b_idx_rand_sh_sec = 0;

  for (const auto& level : circ_.gates_by_level) {
    for (const auto& gate : level) {
      switch (gate->type) {
        case common::utils::GateType::kInp: {
          AuthAddShare<Ring> share_r;
          randomShare(nP_, id_, rgen_, share_r, rand_sh_sec, idx_rand_sh_sec);
          auto pid = input_pid_map.at(gate->out);
          auto pregate = std::make_unique<PreprocInput<Ring>>(pid, share_r);
          preproc_.gates[gate->out] = std::move(pregate);
          break;
        }

        case common::utils::GateType::kRec: {
          auto pregate = std::make_unique<PreprocRecGate<Ring>>();
          // King party (party 1) receives the reconstructed value
          bool is_king = (id_ == 1);
          pregate->Pking = is_king;
          preproc_.gates[gate->out] = std::move(pregate);
          break;
        }

        default: {
          break;
        }
      }
    }
  }
}


void OfflineEvaluator::setWireMasks(const std::unordered_map<common::utils::wire_t, int>& input_pid_map) {
  std::vector<Ring> rand_sh_sec;

  if (id_ == 0) {
    setWireMasksParty(input_pid_map, rand_sh_sec);

    size_t rand_sh_sec_num = rand_sh_sec.size();
    size_t arith_comm = rand_sh_sec_num;
    std::vector<size_t> lengths(2);
    lengths[0] = arith_comm;
    lengths[1] = rand_sh_sec_num;

    network_->send(nP_, lengths.data(), sizeof(size_t) * lengths.size());

    std::vector<Ring> offline_arith_comm(arith_comm);

    for (size_t i = 0; i < rand_sh_sec_num; i++) {
      offline_arith_comm[i] = rand_sh_sec[i];
    }
    network_->send(nP_, offline_arith_comm.data(), sizeof(Ring) * arith_comm);

  } else if (id_ != nP_) {
    setWireMasksParty(input_pid_map, rand_sh_sec);

  } else {

    std::vector<size_t> lengths(2);
    usleep(latency_);
    network_->recv(0, lengths.data(), sizeof(size_t) * lengths.size());
    size_t arith_comm = lengths[0];
    size_t rand_sh_sec_num = lengths[1];

    std::vector<Ring> offline_arith_comm(arith_comm);
    network_->recv(0, offline_arith_comm.data(), sizeof(Ring) * arith_comm);

    rand_sh_sec.resize(rand_sh_sec_num);
    for (int i = 0; i < rand_sh_sec_num; i++) {
      rand_sh_sec[i] = offline_arith_comm[i];
    }

    setWireMasksParty(input_pid_map, rand_sh_sec);
  }
}

PreprocCircuit<Ring> OfflineEvaluator::getPreproc() {
  return std::move(preproc_);
}

PreprocCircuit<Ring> OfflineEvaluator::run(const std::unordered_map<common::utils::wire_t, int>& input_pid_map) {
  setWireMasks(input_pid_map);
  return std::move(preproc_);
}

};  // namespace graphdb