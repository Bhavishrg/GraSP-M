#include "offline_evaluator.h"

#include <NTL/BasicThreadPool.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>

// #include "../utils/helpers.h"

namespace graphdb {
OfflineEvaluator::OfflineEvaluator(int nP, int my_id,
                                   std::shared_ptr<io::NetIOMP> network,
                                   common::utils::LevelOrderedCircuit circ,
                                   int threads, int seed)
    : nP_(nP),
      id_(my_id),
      rgen_(my_id, seed), 
      network_(std::move(network)),
      circ_(std::move(circ))
      // preproc_(circ.num_gates)

      { } // tpool_ = std::make_shared<ThreadPool>(threads); }

void OfflineEvaluator::randomShare(int nP, int pid, RandGenPool& rgen, AddShare<Ring>& share, TPShare<Ring>& tpShare) {
  Ring val = Ring(0);
  if (pid == 0) {
    share.pushValue(Ring(0));
    tpShare.pushValues(Ring(0));
    for (int i = 1; i <= nP; i++) {
      rgen.pi(i).random_data(&val, sizeof(Ring));
      tpShare.pushValues(val);
    }
  } else {
    rgen.p0().random_data(&val, sizeof(Ring));
    share.pushValue(val);
  }
}

void OfflineEvaluator::randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                         AddShare<Ring>& share, TPShare<Ring>& tpShare, Ring secret,
                                         std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec, bool print) {
  if (pid == 0) {
    Ring val = Ring(0);
    Ring valn = Ring(0);
    share.pushValue(Ring(0));
    tpShare.pushValues(Ring(0));
    for (int i = 1; i < nP; i++) {
      rgen.pi(i).random_data(&val, sizeof(Ring));
      tpShare.pushValues(val);
      valn += val;
      if (print){
        std::cout << "Eqz r2 share of p1: " << val << std::endl;
      }
      
    }
    valn = secret - valn;
    tpShare.pushValues(valn);
    rand_sh_sec.push_back(valn);
    if (print){
        std::cout << "Eqz r2 share of p2: " << valn << std::endl;
    }
  } else {
    if (pid != nP) {
      Ring val;
      rgen.p0().random_data(&val, sizeof(Ring));
      share.pushValue(val);
      if (print){
        std::cout << "Eqz r2 share of p" << pid << ": " << val << std::endl;
      }
    } else {
      share.pushValue(rand_sh_sec[idx_rand_sh_sec]);
      idx_rand_sh_sec++;
      if (print){
        std::cout << "Eqz r2 share of p" << pid << ": " << share.valueAt() << std::endl;
      }
    }
  }
}

void OfflineEvaluator::randomPermutation(int nP, int pid, RandGenPool& rgen, std::vector<int>& pi, size_t& vec_size) {
  if (pid != 0) {
    for (int i = 0; i < vec_size; ++i) {
      pi[i] = i;
    }
  }
}

void OfflineEvaluator::generateShuffleDeltaVector(int nP, int pid, RandGenPool& rgen, std::vector<AddShare<Ring>>& delta,
                                                  std::vector<TPShare<Ring>>& tp_a, std::vector<TPShare<Ring>>& tp_b,
                                                  std::vector<TPShare<Ring>>& tp_c, std::vector<std::vector<int>>& tp_pi_all,
                                                  size_t& vec_size, std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {
    std::vector<Ring> deltan(vec_size);
    Ring valn;
    for (int i = 0; i < vec_size; ++i) {
      Ring val_a = tp_a[i].secret() - tp_a[i][1];
      int idx_perm = i;
      for (int j = 0; j < nP; ++j) {
        idx_perm = tp_pi_all[j][idx_perm];
        val_a += tp_c[idx_perm][j + 1];
      }
      Ring val_b = tp_b[idx_perm].secret() - tp_b[idx_perm][nP];
      deltan[idx_perm] = val_a - tp_c[idx_perm][nP] - val_b;
    }
    for (int i = 0; i < vec_size; ++i) {
      rand_sh_sec.push_back(deltan[i]);
    }
  } else if (pid == nP) {
    for (int i = 0; i < vec_size; ++i) {
      delta[i].pushValue(rand_sh_sec[idx_rand_sh_sec]);
      idx_rand_sh_sec++;
    }
  }
}

void OfflineEvaluator::generatePermAndShDeltaVector(int nP, int pid, RandGenPool& rgen, int owner, std::vector<AddShare<Ring>>& delta,
                                                    std::vector<TPShare<Ring>>& tp_a, std::vector<TPShare<Ring>>& tp_b,
                                                    std::vector<int>& pi, size_t& vec_size, std::vector<Ring>& delta_sh, size_t& idx_delta_sh) {
  if (pid == 0) {
    std::vector<Ring> deltan(vec_size);
    for (int i = 0; i < vec_size; ++i) {
      Ring val_a = tp_a[i].secret() - tp_a[i][owner];
      int idx_perm = pi[i];
      Ring val_b = tp_b[idx_perm].secret() - tp_b[idx_perm][owner];
      deltan[idx_perm] = val_a - val_b;
    }
    for (int i = 0; i < vec_size; ++i) {
      delta_sh.push_back(deltan[i]);
    }
  } else if (pid == owner) {
    for (int i = 0; i < vec_size; ++i) {
      delta[i].pushValue(delta_sh[idx_delta_sh]);
      idx_delta_sh++;
    }
  }
}

void OfflineEvaluator::setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map, 
                                         std::vector<Ring>& rand_sh_sec,
                                         std::vector<std::vector<Ring>>& delta_sh) {
  size_t idx_rand_sh_sec = 0;
  size_t idx_delta_sh = 0;
  size_t b_idx_rand_sh_sec = 0;

  for (const auto& level : circ_.gates_by_level) {
    for (const auto& gate : level) {
      switch (gate->type) {
        case common::utils::GateType::kInp: {
          auto pregate = std::make_unique<PreprocInput<Ring>>();
          auto pid = input_pid_map.at(gate->out);
          pregate->pid = pid;
          preproc_.gates[gate->out] = std::move(pregate);
          break;
        }

        case common::utils::GateType::kMul: {
          AddShare<Ring> triple_a; // Holds one beaver triple share of a random value a
          TPShare<Ring> tp_triple_a; // Holds all the beaver triple shares of a random value a
          AddShare<Ring> triple_b; // Holds one beaver triple share of a random value b
          TPShare<Ring> tp_triple_b; // Holds all the beaver triple shares of a random value b
          AddShare<Ring> triple_c; // Holds one beaver triple share of c=a*b
          TPShare<Ring> tp_triple_c; // Holds all the beaver triple shares of c=a*b
          randomShare(nP_, id_, rgen_, triple_a, tp_triple_a);
          randomShare(nP_, id_, rgen_, triple_b, tp_triple_b);
          Ring tp_prod;
          if (id_ == 0) { tp_prod = tp_triple_a.secret() * tp_triple_b.secret(); }
          randomShareSecret(nP_, id_, rgen_, triple_c, tp_triple_c, tp_prod, rand_sh_sec, idx_rand_sh_sec);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocMultGate<Ring>>(triple_a, tp_triple_a, triple_b, tp_triple_b, triple_c, tp_triple_c));
          break;
        }

         case common::utils::GateType::kEqz: {
          AddShare<Ring> share_r1;
          TPShare<Ring> tp_share_r1;
          AddShare<Ring> share_r2;
          TPShare<Ring> tp_share_r2;
          std::vector<AddShare<Ring>> share_r1_bits(RINGSIZEBITS);
          std::vector<TPShare<Ring>> tp_share_r1_bits(RINGSIZEBITS);
          std::vector<AddShare<Ring>> share_r2_bits(RINGSIZEBITS);
          std::vector<TPShare<Ring>> tp_share_r2_bits(RINGSIZEBITS);
          Ring tp_r1 = Ring(0);
          Ring tp_r2 = Ring(0);
          std::vector<Ring> tp_r1_bits(RINGSIZEBITS);
          std::vector<Ring> tp_r2_bits(RINGSIZEBITS);

          // sharing r1 and r1_bits
          randomShare(nP_, id_, rgen_, share_r1, tp_share_r1);
          
          if (id_ == 0) {
            tp_r1 = tp_share_r1.secret();
            tp_r1_bits = bitDecomposeToInt(tp_r1);
          }
          for (int i = 0; i < RINGSIZEBITS; ++i) {
              randomShareSecret(nP_, id_, rgen_, share_r1_bits[i], tp_share_r1_bits[i], tp_r1_bits[i],
                                                    rand_sh_sec, idx_rand_sh_sec);                                      
          }

          // sharing r2 and r2_bits
          if (id_ == 0) {
            rgen_.p0().random_data(&tp_r2, sizeof(Ring));
            tp_r2 = tp_r2 % RINGSIZEBITS; // make sure r2 is in [0, RINGSIZEBITS-1]
          }
          randomShareSecret(nP_, id_, rgen_, share_r2, tp_share_r2, tp_r2, rand_sh_sec, idx_rand_sh_sec, 1);

          if (id_ == 0) {
            tp_r2 = tp_share_r2.secret();
            for (int i = 0; i < RINGSIZEBITS; ++i) {
              if (i == tp_r2 % RINGSIZEBITS) {
                tp_r2_bits[i] = 1;
              } else {
                tp_r2_bits[i] = 0;
              }
            }
          }

          for (int i = 0; i < RINGSIZEBITS; ++i) {
            randomShareSecret(nP_, id_, rgen_, share_r2_bits[i], tp_share_r2_bits[i], tp_r2_bits[i],
                                                    rand_sh_sec, idx_rand_sh_sec);
          }
          preproc_.gates[gate->out] =
              std::make_unique<PreprocEqzGate<Ring>>(share_r1, tp_share_r1, share_r2, tp_share_r2, share_r1_bits, tp_share_r1_bits, share_r2_bits, tp_share_r2_bits);
          break;
        }

        case common::utils::GateType::kShuffle: {
          auto *shuffle_g = static_cast<common::utils::SIMDOGate *>(gate.get());
          auto vec_size = shuffle_g->in.size();
          std::vector<AddShare<Ring>> a(vec_size); // Randomly sampled vector
          std::vector<TPShare<Ring>> tp_a(vec_size); // Randomly sampled vector
          std::vector<AddShare<Ring>> b(vec_size); // Randomly sampled vector
          std::vector<TPShare<Ring>> tp_b(vec_size); // Randomly sampled vector
          std::vector<AddShare<Ring>> c(vec_size); // Randomly sampled vector
          std::vector<TPShare<Ring>> tp_c(vec_size); // Randomly sampled vector
          for (int i = 0; i < vec_size; i++) {
            randomShare(nP_, id_, rgen_, a[i], tp_a[i]);
            randomShare(nP_, id_, rgen_, b[i], tp_b[i]);
            randomShare(nP_, id_, rgen_, c[i], tp_c[i]);
          }

          std::vector<int> pi; // Randomly sampled permutation using HP
          std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutations of all parties using HP
          if (id_ != 0) {
            pi = std::move(shuffle_g->permutation[0]);
          } else {
            tp_pi_all = std::move(shuffle_g->permutation);
          }

          std::vector<int> pi_common(vec_size); // Common random permutation held by all parties except HP. HP holds dummy values
          if (id_ != 0) { randomPermutation(nP_, id_, rgen_, pi_common, vec_size); }

          std::vector<AddShare<Ring>> delta(vec_size); // Delta vector only held by the last party. Dummy values for the other parties
          generateShuffleDeltaVector(nP_, id_, rgen_, delta, tp_a, tp_b, tp_c, tp_pi_all, vec_size, rand_sh_sec, idx_rand_sh_sec);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocShuffleGate<Ring>>(a, tp_a, b, tp_b, c, tp_c, delta, pi, tp_pi_all, pi_common));
          break;
        }

        case common::utils::GateType::kPermAndSh: {
          auto *permAndSh_g = static_cast<common::utils::SIMDOGate *>(gate.get());
          auto vec_size = permAndSh_g->in.size();
          std::vector<AddShare<Ring>> a(vec_size); // Randomly sampled vector
          std::vector<TPShare<Ring>> tp_a(vec_size); // Randomly sampled vector
          std::vector<AddShare<Ring>> b(vec_size); // Randomly sampled vector
          std::vector<TPShare<Ring>> tp_b(vec_size); // Randomly sampled vector
          for (int i = 0; i < vec_size; i++) {
            randomShare(nP_, id_, rgen_, a[i], tp_a[i]);
            randomShare(nP_, id_, rgen_, b[i], tp_b[i]);
          }

          std::vector<int> pi; // Randomly sampled permutation using HP
          std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutation of gate owner party using HP.
          if (id_ != 0) {
            pi = std::move(permAndSh_g->permutation[0]);
          } else {
            tp_pi_all = std::move(permAndSh_g->permutation);
          }

          std::vector<int> pi_common(vec_size); // Common random permutation held by all parties except HP. HP holds dummy values
          if (id_ != 0) { randomPermutation(nP_, id_, rgen_, pi_common, vec_size); }

          std::vector<AddShare<Ring>> delta(vec_size); // Delta vector only held by the gate owner party. Dummy values for the other parties
          generatePermAndShDeltaVector(nP_, id_, rgen_, gate->owner, delta, tp_a, tp_b,
                                       tp_pi_all[gate->owner - 1], vec_size, delta_sh[gate->owner - 1], idx_delta_sh);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocPermAndShGate<Ring>>(a, tp_a, b, tp_b, delta, pi, tp_pi_all, pi_common));
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
  std::vector<std::vector<Ring>> delta_sh(nP_, std::vector<Ring>());

  if (id_ == 0) {
    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh);

    for (int pid = 1; pid < nP_; ++pid) {
      size_t delta_sh_num = delta_sh[pid - 1].size();
      network_->send(pid, &delta_sh_num, sizeof(size_t));
      network_->send(pid, delta_sh[pid - 1].data(), delta_sh_num * sizeof(size_t));
    }

    size_t rand_sh_sec_num = rand_sh_sec.size();
    size_t delta_sh_last_num = delta_sh[nP_ - 1].size();
    size_t arith_comm = rand_sh_sec_num;
    std::vector<size_t> lengths(3);
    lengths[0] = arith_comm;
    lengths[1] = rand_sh_sec_num;
    lengths[2] = delta_sh_last_num;

    network_->send(nP_, lengths.data(), sizeof(size_t) * lengths.size());

    std::vector<Ring> offline_arith_comm(arith_comm);

    for (size_t i = 0; i < rand_sh_sec_num; i++) {
      offline_arith_comm[i] = rand_sh_sec[i];
    }
    network_->send(nP_, offline_arith_comm.data(), sizeof(Ring) * arith_comm);
    network_->send(nP_, delta_sh[nP_ - 1].data(), sizeof(Ring) * delta_sh_last_num);

  } else if (id_ != nP_) {

    size_t delta_sh_num;
    usleep(250);
    network_->recv(0, &delta_sh_num, sizeof(size_t));
    std::vector<std::vector<Ring>> delta_sh(nP_);
    delta_sh[id_ - 1] = std::vector<Ring>(delta_sh_num);
    network_->recv(0, delta_sh[id_ - 1].data(), delta_sh_num * sizeof(Ring));
    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh);

  } else {

    std::vector<size_t> lengths(3);
    usleep(250);
    network_->recv(0, lengths.data(), sizeof(size_t) * lengths.size());
    size_t arith_comm = lengths[0];
    size_t rand_sh_sec_num = lengths[1];
    size_t delta_sh_num = lengths[2];

    std::vector<Ring> offline_arith_comm(arith_comm);
    network_->recv(0, offline_arith_comm.data(), sizeof(Ring) * arith_comm);

    std::vector<std::vector<Ring>> delta_sh(nP_);
    delta_sh[id_ - 1] = std::vector<Ring>(delta_sh_num);
    network_->recv(0, delta_sh[id_ - 1].data(), sizeof(Ring) * delta_sh_num);

    rand_sh_sec.resize(rand_sh_sec_num);
    for (int i = 0; i < rand_sh_sec_num; i++) {
      rand_sh_sec[i] = offline_arith_comm[i];
    }

    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh);
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
