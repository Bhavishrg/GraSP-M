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
      rgen_(my_id, nP, seed), 
      network_(std::move(network)),
      circ_(std::move(circ)) 
      
      {} // tpool_ = std::make_shared<ThreadPool>(threads); }


void OfflineEvaluator::randomShare(int nP, int pid, RandGenPool& rgen, AuthAddShare& share,
                                         std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
    
  if (pid == 0) {
    Field val = Field(0);
    Field valn = Field(0);
    Field tag = Field(0);
    Field tagn = Field(0);
    Field key = share.keySh();

    for (int i = 1; i <= nP; i++) {
        randomizeZZp(rgen.pi(i), val, sizeof(Field));
        valn += val;
    }
    
    tagn = key * valn;
    share.pushValue(valn);
    share.pushTag(tagn);

    for (int i = 1; i < nP; i++) {
       randomizeZZp(rgen.pi(i), tag, sizeof(Field));
        tagn -= tag;
    }
    rand_sh_sec.push_back(tagn);

  } else {
    Field val;
    Field tag;
    if (pid != nP) {
        randomizeZZp(rgen.p0(), val, sizeof(Field));
        randomizeZZp(rgen.p0(), tag, sizeof(Field));
        share.pushValue(val);
        share.pushTag(tag);
    } else {
        randomizeZZp(rgen.p0(), val, sizeof(Field));
        share.pushValue(val);
        share.pushTag(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
    }
  }
}

void OfflineEvaluator::randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                         AuthAddShare& share, Field secret,
                                         std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {

    Field key = share.keySh();
    Field val = Field(0);
    Field valn = secret;

    Field tag = Field(0);
    Field tagn = key * secret;

    share.pushValue(valn);
    share.pushTag(tagn);

    

    for (int i = 1; i < nP; i++) {

        randomizeZZp(rgen.pi(i), val, sizeof(Field));
        randomizeZZp(rgen.pi(i), tag, sizeof(Field));
        valn -= val;
        tagn -= tag;
    }
    
    rand_sh_sec.push_back(valn);
    rand_sh_sec.push_back(tagn);
    

  } else {
    Field val;
    Field tag;
    if (pid != nP) {

        randomizeZZp(rgen.p0(), val, sizeof(Field));
        randomizeZZp(rgen.p0(), tag, sizeof(Field));
        share.pushValue(val);
        share.pushTag(tag);

    } else {

        share.pushValue(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
        share.pushTag(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
        
    }
  }
}

void OfflineEvaluator::generatePermAndShPermutedMask(int nP, int pid, RandGenPool& rgen,
                                                    std::vector<AuthAddShare>& mask_R,
                                                    std::vector<AuthAddShare>& permuted_mask,
                                                    std::vector<AuthAddShare>& mask_R_tag,
                                                    std::vector<AuthAddShare>& permuted_mask_tag,
                                                    std::vector<int>& owner_pi,
                                                    size_t vec_size, int owner,
                                                    std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {
    // Reconstruct R from shares locally

    // Compute π_owner(R)
    std::vector<Field> permuted_R(vec_size);
    std::vector<Field> permuted_R_tag(vec_size);
    for (size_t j = 0; j < vec_size; j++) {
      int idx_perm = owner_pi[j];
      permuted_R[idx_perm] = mask_R[j].valueAt();
      permuted_R_tag[idx_perm] = mask_R_tag[j].valueAt();
    }

    // Share π_owner(R) using randomShareSecret
    for (size_t j = 0; j < vec_size; j++) {
      randomShareSecret(nP, pid, rgen, permuted_mask[j], permuted_R[j], rand_sh_sec, idx_rand_sh_sec);
      randomShareSecret(nP, pid, rgen, permuted_mask_tag[j], permuted_R_tag[j], rand_sh_sec, idx_rand_sh_sec);
    }
    
  } else {
    // Computing parties receive shares via randomShareSecret
    for (size_t j = 0; j < vec_size; j++) {
      randomShareSecret(nP, pid, rgen, permuted_mask[j], Field(0), rand_sh_sec, idx_rand_sh_sec);
      randomShareSecret(nP, pid, rgen, permuted_mask_tag[j], Field(0), rand_sh_sec, idx_rand_sh_sec);
    }
  }
}

// Generate shares of permuted masks π_i(R) for all parties i=1 to nP
// HP: Reconstructs R, computes π_i(R) for each party, then uses randomShareSecret to share each π_i(R)
// Computing parties: Receive their shares via randomShareSecret
void OfflineEvaluator::generateAmortzdPnSPermutedMasks(int nP, int pid, RandGenPool& rgen,
                                                        std::vector<AuthAddShare> mask_R,
                                                        std::vector<std::vector<AuthAddShare>> permuted_masks,
                                                        std::vector<AuthAddShare> mask_R_tag,
                                                        std::vector<std::vector<AuthAddShare>> permuted_masks_tag,
                                                        std::vector<std::vector<int>>& all_permutations,
                                                        size_t vec_size,
                                                        std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {
    // Compute π_i(R) for i = 0 to nP-1
    for (size_t party = 0; party < nP; party++) {
      std::vector<int>& pi_i = all_permutations[party];
      std::vector<Field> permuted_R(vec_size);
      std::vector<Field> permuted_R_tag(vec_size);

      // Compute π_i(R)
      for (size_t j = 0; j < vec_size; j++) {
        int idx_perm = pi_i[j];
        permuted_R[idx_perm] = mask_R[j].valueAt();
        permuted_R_tag[idx_perm] = mask_R_tag[j].valueAt();
      }

      // Share π_i(R) using randomShareSecret
      for (size_t j = 0; j < vec_size; j++) {
        randomShareSecret(nP, pid, rgen, permuted_masks[party][j], permuted_R[j], rand_sh_sec, idx_rand_sh_sec);
        randomShareSecret(nP, pid, rgen, permuted_masks_tag[party][j], permuted_R_tag[j], rand_sh_sec, idx_rand_sh_sec);
      }
    }
  } else {
    // Computing parties: Receive shares via randomShareSecret
    for (size_t party = 0; party < nP; party++) {
      for (size_t j = 0; j < vec_size; j++) {
        randomShareSecret(nP, pid, rgen, permuted_masks[party][j], Field(0), rand_sh_sec, idx_rand_sh_sec);
        randomShareSecret(nP, pid, rgen, permuted_masks_tag[party][j], Field(0), rand_sh_sec, idx_rand_sh_sec);
      }
    }
  }
}


void OfflineEvaluator::setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map, 
                                         std::vector<Field>& rand_sh_sec)  {
  size_t idx_rand_sh_sec = 0;
  size_t b_idx_rand_sh_sec = 0;

  for (const auto& level : circ_.gates_by_level) {
    for (const auto& gate : level) {
      switch (gate->type) {

        case common::utils::GateType::kInp: {
            auto pid = input_pid_map.at(gate->out);

            Field r = Field(0);

            // Sample the mask value
            if (id_ == 0) {
              randomizeZZp(rgen_.pi(pid), r, sizeof(Field));
            }
            else if (pid == id_) {
              randomizeZZp(rgen_.p0(), r, sizeof(Field));
            }            
            // Generate authenticated shares of the mask
            AuthAddShare r_sh;
            randomShareSecret(nP_, id_, rgen_, r_sh, r, rand_sh_sec, idx_rand_sh_sec);

            auto pregate = std::make_unique<PreprocInput>(pid, r_sh, r);
            preproc_.gates[gate->out] = std::move(pregate);
            break;
        }

        case common::utils::GateType::kRec: {
          auto pregate = std::make_unique<PreprocRecGate>();
          // King party (party 1) receives the reconstructed value
          bool is_king = (id_ == 1);
          pregate->Pking = is_king;
          preproc_.gates[gate->out] = std::move(pregate);
          break;
        }

        case common::utils::GateType::kMul: {

          AuthAddShare triple_a; // Holds one beaver triple share of a random value a
          AuthAddShare triple_b; // Holds one beaver triple share of a random value b
          AuthAddShare triple_c; // Holds one beaver triple share of c=a*b

          randomShare(nP_, id_, rgen_, triple_a, rand_sh_sec, idx_rand_sh_sec);
          randomShare(nP_, id_, rgen_, triple_b, rand_sh_sec, idx_rand_sh_sec);

          Field tp_prod;
          if (id_ == 0) { tp_prod = triple_a.valueAt() * triple_b.valueAt(); }
          randomShareSecret(nP_, id_, rgen_, triple_c, tp_prod, rand_sh_sec, idx_rand_sh_sec);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocMultGate>(triple_a, triple_b, triple_c));
          break;
        }


        case common::utils::GateType::kEqz: {
          AuthAddShare share_r1;
          AuthAddShare share_r2;
          std::vector<AuthAddShare> share_r1_bits(RINGSIZEBITS);
          std::vector<AuthAddShare> tp_share_r1_bits(RINGSIZEBITS);
          std::vector<AuthAddShare> share_r2_bits(RINGSIZEBITS);
          std::vector<AuthAddShare> tp_share_r2_bits(RINGSIZEBITS);
          Field tp_r1 = Field(0);
          Field tp_r2 = Field(0);
          std::vector<Field> tp_r1_bits(RINGSIZEBITS);
          std::vector<Field> tp_r2_bits(RINGSIZEBITS);

          // sharing r1 and r1_bits
          randomShare(nP_, id_, rgen_, share_r1, rand_sh_sec, idx_rand_sh_sec);
          
          if (id_ == 0) {
            tp_r1 = share_r1.valueAt();
            uint64_t tp_r1_uint = conv<uint64_t>(tp_r1);
            auto num_bits = sizeof(tp_r1_uint) * 8;
            for (size_t i = 0; i < num_bits; ++i) {
              tp_r1_bits[i] = ((tp_r1_uint >> i) & 1ULL);
            }
          }
          for (int i = 0; i < RINGSIZEBITS; ++i) {
              randomShareSecret(nP_, id_, rgen_, share_r1_bits[i], tp_r1_bits[i],
                                                    rand_sh_sec, idx_rand_sh_sec);                                      
          }

          // sharing r2 and r2_bits
          uint64_t tp_r2_uint = 0;
          if (id_ == 0) {
            randomizeZZp(rgen_.p0(), tp_r2, sizeof(Field));
            tp_r2_uint = conv<uint64_t>(tp_r2) % RINGSIZEBITS; // make sure r2 is in [0, RINGSIZEBITS-1]
          }
          randomShareSecret(nP_, id_, rgen_, share_r2, Field(tp_r2_uint), rand_sh_sec, idx_rand_sh_sec);

          if (id_ == 0) {
            for (int i = 0; i < RINGSIZEBITS; ++i) {
              if (i == tp_r2_uint) {
                tp_r2_bits[i] = Field(1);
              } else {
                tp_r2_bits[i] = Field(0);
              }
            }
          }

          for (int i = 0; i < RINGSIZEBITS; ++i) {
            randomShareSecret(nP_, id_, rgen_, share_r2_bits[i], tp_r2_bits[i],
                                                    rand_sh_sec, idx_rand_sh_sec);
          }
          preproc_.gates[gate->out] =
              std::make_unique<PreprocEqzGate>(share_r1, share_r2, share_r1_bits, share_r2_bits);
          break;
        }

        case common::utils::GateType::kPermAndSh: {
          // Reimplemented: generate mask R and permuted mask π_owner(R)
          auto *permAndSh_g = static_cast<common::utils::SIMDOGate *>(gate.get());
          auto vec_size = permAndSh_g->in.size();

          std::vector<AuthAddShare> mask_R(vec_size);
          std::vector<AuthAddShare> mask_R_tag(vec_size);
          
          for (size_t i = 0; i < vec_size; ++i) {
            randomShare(nP_, id_, rgen_, mask_R[i], rand_sh_sec, idx_rand_sh_sec);
            randomShare(nP_, id_, rgen_, mask_R_tag[i], rand_sh_sec, idx_rand_sh_sec);
          }

          // Generate permuted mask π_owner(R) via helper using randomShareSecret
          std::vector<AuthAddShare> permuted_mask(vec_size);
          std::vector<AuthAddShare> permuted_mask_tag(vec_size);
          // Owner permutation is available in the gate for all parties (including helper)
          std::vector<int> owner_pi = permAndSh_g->permutation[0];

          generatePermAndShPermutedMask(nP_, id_, rgen_, mask_R, permuted_mask, mask_R_tag, permuted_mask_tag,
                                        owner_pi, vec_size, gate->owner, rand_sh_sec, idx_rand_sh_sec);

          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocPermAndShGate>(mask_R, permuted_mask, mask_R_tag, permuted_mask_tag, vec_size, gate->owner));
          break;
        }


        case common::utils::GateType::kAmortzdPnS: {
          auto *amortzdPnS_g = static_cast<common::utils::SIMDOGate *>(gate.get());
          auto vec_size = amortzdPnS_g->vec_size;  // Use metadata stored in gate
          
          // Generate random mask R and its shares
          std::vector<AuthAddShare> mask_R(vec_size);
          std::vector<AuthAddShare> mask_R_tag(vec_size);

          for (size_t i = 0; i < vec_size; i++) {
            randomShare(nP_, id_, rgen_, mask_R[i], rand_sh_sec, idx_rand_sh_sec);
            randomShare(nP_, id_, rgen_, mask_R_tag[i], rand_sh_sec, idx_rand_sh_sec);
          }
          
          // Generate shares of permuted masks π_i(R) for each party i
          std::vector<std::vector<AuthAddShare>> permuted_masks(nP_, std::vector<AuthAddShare>(vec_size));
          std::vector<std::vector<AuthAddShare>> permuted_masks_tag(nP_, std::vector<AuthAddShare>(vec_size));
          
          // HP computes π_i(R) for each party i and shares them using randomShareSecret
          generateAmortzdPnSPermutedMasks(nP_, id_, rgen_, mask_R, permuted_masks,
                                          mask_R_tag, permuted_masks_tag,
                                          amortzdPnS_g->permutation, vec_size,
                                          rand_sh_sec, idx_rand_sh_sec);
          
          preproc_.gates[gate->outs[0]] = std::move(std::make_unique<PreprocAmortzdPnSGate>(mask_R, permuted_masks, 
                                                                mask_R_tag, permuted_masks_tag, 
                                                                vec_size, nP_));
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
  std::vector<Field> rand_sh_sec;

  if (id_ == 0) {
    setWireMasksParty(input_pid_map, rand_sh_sec);
    size_t rand_sh_sec_num = rand_sh_sec.size();
    size_t arith_comm = rand_sh_sec_num;
    std::vector<size_t> lengths(2);
    lengths[0] = arith_comm;
    lengths[1] = rand_sh_sec_num;

    network_->send(nP_, lengths.data(), sizeof(size_t) * lengths.size());

    std::vector<Field> offline_arith_comm(arith_comm);

    for (size_t i = 0; i < rand_sh_sec_num; i++) {
      offline_arith_comm[i] = rand_sh_sec[i];
    }
    network_->send(nP_, offline_arith_comm.data(), sizeof(Field) * arith_comm);

  } else if (id_ != nP_) {
    setWireMasksParty(input_pid_map, rand_sh_sec);

  } else {

    std::vector<size_t> lengths(2);
    usleep(latency_);
    network_->recv(0, lengths.data(), sizeof(size_t) * lengths.size());
    size_t arith_comm = lengths[0];
    size_t rand_sh_sec_num = lengths[1];

    std::vector<Field> offline_arith_comm(arith_comm);
    network_->recv(0, offline_arith_comm.data(), sizeof(Field) * arith_comm);

    rand_sh_sec.resize(rand_sh_sec_num);
    for (int i = 0; i < rand_sh_sec_num; i++) {
      rand_sh_sec[i] = offline_arith_comm[i];
    }

    setWireMasksParty(input_pid_map, rand_sh_sec);
  }
}

PreprocCircuit OfflineEvaluator::getPreproc() {
  return std::move(preproc_);
}

PreprocCircuit OfflineEvaluator::run(const std::unordered_map<common::utils::wire_t, int>& input_pid_map) {
  initializeGlobalKey(nP_, id_, rgen_, network_);

  setWireMasks(input_pid_map);

  return std::move(preproc_);
}

};  // namespace graphdb