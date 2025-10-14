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
  // Generate common random permutation between party 0 and party pid
  if (pid != 0) {
    pi.resize(vec_size);
    for (int i = 0; i < vec_size; ++i) {
      pi[i] = i;
    }
    // Fisher-Yates shuffle using common randomness with party 0
    for (int i = vec_size - 1; i > 0; --i) {
      uint32_t rand_val;
      rgen.p0().random_data(&rand_val, sizeof(uint32_t));
      int j = rand_val % (i + 1);
      std::swap(pi[i], pi[j]);
    }
  }
}

void OfflineEvaluator::generateShuffleDeltaVector(int nP, int pid, RandGenPool& rgen, std::vector<Ring>& delta,
                                                  std::vector<TPShare<Ring>>& tp_a, std::vector<TPShare<Ring>>& tp_b,
                                                  std::vector<TPShare<Ring>>& tp_c, std::vector<std::vector<int>>& tp_pi_all,
                                                  size_t& vec_size, std::vector<Ring>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {
    // Party 0 generates Δ for all shuffle gates and stores it in array
    // Δ = Π(a) + Σᵢ₌1 to ⁿ Πn·Πᵢ₋₁···Π₁(bn-ᵢ+1) + c
    // where Πᵢ is the permutation of party i
    // Π = Πₙ·Πₙ₋₁···Π₁ (composition of all party permutations)
    std::vector<Ring> deltan(vec_size);
    
    for (int i = 0; i < vec_size; ++i) {
      // Start with index i and apply all permutations from party 1 to nP: Π = Πₙ·Πₙ₋₁···Π₁
      int idx_perm = i;
      for (int j = 0; j < nP; ++j) {
        idx_perm = tp_pi_all[j][idx_perm];
      }
      
      // Compute Π(a[i])
      Ring pi_a = tp_a[idx_perm].secret();
      
      // Compute Σᵢ₌₁ⁿ Πn·Πᵢ₋₁···Π₁(bn-ᵢ+1)
      // For i from 1 to n:
      //   - Apply Π₁, Π₂, ..., Πᵢ₋₁ to position i to get intermediate index
      //   - Get b[intermediate_idx] for party (n-i+1)
      //   - Apply Πᵢ, Πᵢ₊₁, ..., Πn to intermediate_idx to get final position
      Ring sum_term = Ring(0);
      for (int i_term = 1; i_term <= nP; ++i_term) {
        // Party ID for this term (reversed: n-i+1)
        int party_id = nP - i_term + 1;
        
        // Apply Π₁, Π₂, ..., Πᵢ₋₁ to starting index i
        int intermediate_idx = i;
        for (int j = 0; j < i_term - 1; ++j) {
          intermediate_idx = tp_pi_all[j][intermediate_idx];
        }
        
        // Get b value for party_id at this intermediate position
        Ring b_value = tp_b[intermediate_idx][party_id];
        
        // Apply Πᵢ, Πᵢ₊₁, ..., Πn to find where this contributes in final output
        int final_idx = intermediate_idx;
        for (int j = i_term - 1; j < nP; ++j) {
          final_idx = tp_pi_all[j][final_idx];
        }
        
        // Accumulate at the final position
        sum_term += b_value;
      }
      
      // Compute c at final permuted position
      Ring c_final = tp_c[idx_perm].secret();
      
      // Final delta: Δ = Π(a) + Σᵢ₌₁ⁿ Πn·Πᵢ₋₁···Π₁(bn-ᵢ+1) - c
      deltan[idx_perm] = pi_a + sum_term + c_final;
    }    
    
    for (int i = 0; i < vec_size; ++i) {
      rand_sh_sec.push_back(deltan[i]);
    }
  } else if (pid == nP) {
    // Last party receives delta values from party 0
    delta.resize(vec_size);
    for (int i = 0; i < vec_size; ++i) {
      delta[i] = rand_sh_sec[idx_rand_sh_sec];
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

        case common::utils::GateType::kRec: {
          auto pregate = std::make_unique<PreprocRecGate<Ring>>();
          // King party (party 1) receives the reconstructed value
          bool is_king = (id_ == 1);
          bool via_pking = true;  // Default: reconstruction via king party
          pregate->Pking = is_king;
          pregate->viaPking = via_pking;
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
              std::move(std::make_unique<PreprocMultGate<Ring>>(triple_a, tp_triple_a, triple_b, tp_triple_b, triple_c, tp_triple_c, use_pking_));
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

          std::vector<Ring> delta(vec_size); // Delta vector only held by the last party. Dummy values for the other parties
          generateShuffleDeltaVector(nP_, id_, rgen_, delta, tp_a, tp_b, tp_c, tp_pi_all, vec_size, rand_sh_sec, idx_rand_sh_sec);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocShuffleGate<Ring>>(a, tp_a, b, tp_b, c, tp_c, delta, pi, tp_pi_all));
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

        case common::utils::GateType::kCompact: {
          // Compact gate preprocessing: shuffle + multiplications
          auto *compact_g = static_cast<common::utils::SIMDOGate *>(gate.get());
          // Input is [t0,...,tn, p1_0,...,p1_n, p2_0,...,p2_n, ...]
          // Output is [t_compact0,...,t_compactn, p1_compact0,...,p1_compactn, p2_compact0,...,p2_compactn, ...]
          auto total_size = compact_g->in.size();
          auto output_size = compact_g->outs.size();
          // vec_size * (1 + num_payloads) = total_size
          // Determine vec_size from permutation size (which is always vec_size)
          auto vec_size = compact_g->permutation[0].size();
          
          // Preprocessing for 3 shuffle operations (p, t, label) - we'll use same shuffle data for all 3
          std::vector<AddShare<Ring>> shuffle_a(vec_size);
          std::vector<TPShare<Ring>> shuffle_tp_a(vec_size);
          std::vector<AddShare<Ring>> shuffle_b(vec_size);
          std::vector<TPShare<Ring>> shuffle_tp_b(vec_size);
          std::vector<AddShare<Ring>> shuffle_c(vec_size);
          std::vector<TPShare<Ring>> shuffle_tp_c(vec_size);
          
          for (int i = 0; i < vec_size; i++) {
            randomShare(nP_, id_, rgen_, shuffle_a[i], shuffle_tp_a[i]);
            randomShare(nP_, id_, rgen_, shuffle_b[i], shuffle_tp_b[i]);
            randomShare(nP_, id_, rgen_, shuffle_c[i], shuffle_tp_c[i]);
          }
          
          std::vector<int> shuffle_pi;
          std::vector<std::vector<int>> shuffle_tp_pi_all;
          if (id_ != 0) {
            shuffle_pi = std::move(compact_g->permutation[0]);
          } else {
            shuffle_tp_pi_all = std::move(compact_g->permutation);
          }
          
          std::vector<Ring> shuffle_delta(vec_size);
          generateShuffleDeltaVector(nP_, id_, rgen_, shuffle_delta, shuffle_tp_a, shuffle_tp_b, shuffle_tp_c,
                                    shuffle_tp_pi_all, vec_size, rand_sh_sec, idx_rand_sh_sec);
          
          // Preprocessing for vec_size multiplications (for label computation)
          std::vector<AddShare<Ring>> mult_triple_a(vec_size);
          std::vector<TPShare<Ring>> mult_tp_triple_a(vec_size);
          std::vector<AddShare<Ring>> mult_triple_b(vec_size);
          std::vector<TPShare<Ring>> mult_tp_triple_b(vec_size);
          std::vector<AddShare<Ring>> mult_triple_c(vec_size);
          std::vector<TPShare<Ring>> mult_tp_triple_c(vec_size);
          
          for (int i = 0; i < vec_size; i++) {
            randomShare(nP_, id_, rgen_, mult_triple_a[i], mult_tp_triple_a[i]);
            randomShare(nP_, id_, rgen_, mult_triple_b[i], mult_tp_triple_b[i]);
            Ring tp_prod;
            if (id_ == 0) { tp_prod = mult_tp_triple_a[i].secret() * mult_tp_triple_b[i].secret(); }
            randomShareSecret(nP_, id_, rgen_, mult_triple_c[i], mult_tp_triple_c[i], tp_prod, rand_sh_sec, idx_rand_sh_sec);
          }
          
          preproc_.gates[gate->out] = std::move(std::make_unique<PreprocCompactGate<Ring>>(
              shuffle_a, shuffle_tp_a, shuffle_b, shuffle_tp_b, shuffle_c, shuffle_tp_c,
              shuffle_delta, shuffle_pi, shuffle_tp_pi_all,
              mult_triple_a, mult_tp_triple_a, mult_triple_b, mult_tp_triple_b,
              mult_triple_c, mult_tp_triple_c, true));
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
  std::vector<std::vector<Ring>> delta_sh_vec(nP_, std::vector<Ring>());

  if (id_ == 0) {
    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh_vec);

    for (int pid = 1; pid < nP_; ++pid) {
      size_t delta_sh_num = delta_sh_vec[pid - 1].size();
      network_->send(pid, &delta_sh_num, sizeof(size_t));
      network_->send(pid, delta_sh_vec[pid - 1].data(), delta_sh_num * sizeof(size_t));
    }

    size_t rand_sh_sec_num = rand_sh_sec.size();
    size_t delta_sh_last_num = delta_sh_vec[nP_ - 1].size();
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
    network_->send(nP_, delta_sh_vec[nP_ - 1].data(), sizeof(Ring) * delta_sh_last_num);

  } else if (id_ != nP_) {

    size_t delta_sh_num;
    usleep(latency_);
    network_->recv(0, &delta_sh_num, sizeof(size_t));
    std::vector<std::vector<Ring>> delta_sh_vec(nP_);
    delta_sh_vec[id_ - 1] = std::vector<Ring>(delta_sh_num);
    network_->recv(0, delta_sh_vec[id_ - 1].data(), delta_sh_num * sizeof(Ring));
    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh_vec);

  } else {

    std::vector<size_t> lengths(3);
    usleep(latency_);
    network_->recv(0, lengths.data(), sizeof(size_t) * lengths.size());
    size_t arith_comm = lengths[0];
    size_t rand_sh_sec_num = lengths[1];
    size_t delta_sh_num = lengths[2];

    std::vector<Ring> offline_arith_comm(arith_comm);
    network_->recv(0, offline_arith_comm.data(), sizeof(Ring) * arith_comm);

    std::vector<std::vector<Ring>> delta_sh_vec(nP_);
    delta_sh_vec[id_ - 1] = std::vector<Ring>(delta_sh_num);
    network_->recv(0, delta_sh_vec[id_ - 1].data(), sizeof(Ring) * delta_sh_num);

    rand_sh_sec.resize(rand_sh_sec_num);
    for (int i = 0; i < rand_sh_sec_num; i++) {
      rand_sh_sec[i] = offline_arith_comm[i];
    }

    setWireMasksParty(input_pid_map, rand_sh_sec, delta_sh_vec);
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
