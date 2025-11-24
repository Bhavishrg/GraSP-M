#pragma once

#include "../utils/circuit.h"
#include "sharing.h"
#include "../utils/types.h"
#include <unordered_map>

using namespace common::utils;

namespace graphdb {
// Preprocessed data for a gate.
template <class R>
struct PreprocGate {
  PreprocGate() = default;
  virtual ~PreprocGate() = default;
};

template <class R>
using preprocg_ptr_t = std::unique_ptr<PreprocGate<R>>;

template <class R>
struct PreprocInput : public PreprocGate<R> {
  // ID of party providing input on wire.
  int pid{};
  PreprocInput() = default;
  PreprocInput(int pid) 
      : PreprocGate<R>(), pid(pid) {}
  PreprocInput(const PreprocInput<R>& pregate) 
      : PreprocGate<R>(), pid(pregate.pid) {}
};

template <class R>
struct PreprocRecGate : public PreprocGate<R> {
  bool Pking = false;
  PreprocRecGate() = default;
  PreprocRecGate(bool Pking)
    : PreprocGate<R>(), Pking(Pking) {}
};

template <class R>
struct PreprocMultGate : public PreprocGate<R> {
  // Secret shared product of inputs masks.
  AddShare<R> triple_a; // Holds one beaver triple share of a random value a
  TPShare<R> tp_triple_a; // Holds all the beaver triple shares of a random value a
  AddShare<R> triple_b; // Holds one beaver triple share of a random value b
  TPShare<R> tp_triple_b; // Holds all the beaver triple shares of a random value b
  AddShare<R> triple_c; // Holds one beaver triple share of c=a*b
  TPShare<R> tp_triple_c; // Holds all the beaver triple shares of c=a*b
  PreprocMultGate() = default;
  PreprocMultGate(const AddShare<R>& triple_a, const TPShare<R>& tp_triple_a,
                  const AddShare<R>& triple_b, const TPShare<R>& tp_triple_b,
                  const AddShare<R>& triple_c, const TPShare<R>& tp_triple_c)
      : PreprocGate<R>(), triple_a(triple_a), tp_triple_a(tp_triple_a),
        triple_b(triple_b), tp_triple_b(tp_triple_b),
        triple_c(triple_c), tp_triple_c(tp_triple_c) {}
};

template <class R>
struct PreprocEqzGate : public PreprocGate<R> {
  AddShare<R> share_r1;
  TPShare<R> tp_share_r1;
  AddShare<R> share_r2;
  TPShare<R> tp_share_r2;
  std::vector<AddShare<R>> share_r1_bits;
  std::vector<TPShare<R>> tp_share_r1_bits;
  std::vector<AddShare<R>> share_r2_bits;
  std::vector<TPShare<R>> tp_share_r2_bits;
  PreprocEqzGate() = default;
  PreprocEqzGate(const AddShare<R> &share_r1, const TPShare<R> &tp_share_r1,
                 const AddShare<R> &share_r2, const TPShare<R> &tp_share_r2,
                 const std::vector<AddShare<R>> &share_r1_bits, const std::vector<TPShare<R>> &tp_share_r1_bits,
                 const std::vector<AddShare<R>> &share_r2_bits, const std::vector<TPShare<R>> &tp_share_r2_bits)
    : PreprocGate<R>(), share_r1(share_r1), tp_share_r1(tp_share_r1), share_r2(share_r2), tp_share_r2(tp_share_r2), share_r1_bits(share_r1_bits), tp_share_r1_bits(tp_share_r1_bits), share_r2_bits(share_r2_bits), tp_share_r2_bits(tp_share_r2_bits) {}
};

template <class R>
struct PreprocShuffleGate : public PreprocGate<R> {
  std::vector<AddShare<R>> a; // Randomly sampled vector
  std::vector<TPShare<R>> tp_a; // Randomly sampled vector
  std::vector<AddShare<R>> b; // Randomly sampled vector
  std::vector<TPShare<R>> tp_b; // Randomly sampled vector
  std::vector<AddShare<R>> c; // Randomly sampled vector
  std::vector<TPShare<R>> tp_c; // Randomly sampled vector
  std::vector<Ring> delta; // Delta vector only held by the last party. Dummy values for the other parties
  std::vector<int> pi; // Randomly sampled permutation using HP
  std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutations of all parties using HP

  PreprocShuffleGate() = default;
  PreprocShuffleGate(const std::vector<AddShare<R>>& a, const std::vector<TPShare<R>>& tp_a,
                     const std::vector<AddShare<R>>& b, const std::vector<TPShare<R>>& tp_b,
                     const std::vector<AddShare<R>>& c, const std::vector<TPShare<R>>& tp_c,
                     const std::vector<R>& delta, const std::vector<int>& pi, const std::vector<std::vector<int>>& tp_pi_all)
      : PreprocGate<R>(), a(a), tp_a(tp_a), b(b), tp_b(tp_b), c(c), tp_c(tp_c), delta(delta),
        pi(pi), tp_pi_all(tp_pi_all) {}
};

template <class R>
struct PreprocPermAndShGate : public PreprocGate<R> {
  // Mask R and its shares
  std::vector<AddShare<R>> mask_R;      // This party's additive shares of R
  std::vector<TPShare<R>> tp_mask_R;    // All parties' shares of R

  // Permuted mask π_owner(R) (only one permutation per gate - the owner's)
  std::vector<AddShare<R>> permuted_mask;      // This party's additive shares of π_owner(R)
  std::vector<TPShare<R>> tp_permuted_mask;    // All parties' shares of π_owner(R)

  size_t vec_size; // Size of input vector
  int owner;       // Owner party id for this permute-and-share gate

  PreprocPermAndShGate() = default;
  PreprocPermAndShGate(const std::vector<AddShare<R>>& mask_R,
                       const std::vector<TPShare<R>>& tp_mask_R,
                       const std::vector<AddShare<R>>& permuted_mask,
                       const std::vector<TPShare<R>>& tp_permuted_mask,
                       size_t vec_size, int owner)
      : PreprocGate<R>(), mask_R(mask_R), tp_mask_R(tp_mask_R), permuted_mask(permuted_mask), tp_permuted_mask(tp_permuted_mask), vec_size(vec_size), owner(owner) {}
};


template <class R>
struct PreprocAmortzdPnSGate : public PreprocGate<R> {
  // Shares of random mask R (size = vec_size)
  std::vector<AddShare<R>> mask_R;      // This party's additive shares of R
  std::vector<TPShare<R>> tp_mask_R;    // All parties' shares of R
  
  // Shares of permuted masks π_i(R) for each party i (size = nP x vec_size)
  // permuted_masks[i] contains shares of π_i(R)
  std::vector<std::vector<AddShare<R>>> permuted_masks;      // This party's additive shares of π_i(R)
  std::vector<std::vector<TPShare<R>>> tp_permuted_masks;    // All parties' shares of π_i(R)
  
  size_t vec_size;  // Size of input vector
  size_t nP;        // Number of parties
  
  PreprocAmortzdPnSGate() = default;
  PreprocAmortzdPnSGate(const std::vector<AddShare<R>>& mask_R,
                        const std::vector<TPShare<R>>& tp_mask_R,
                        const std::vector<std::vector<AddShare<R>>>& permuted_masks,
                        const std::vector<std::vector<TPShare<R>>>& tp_permuted_masks,
                        size_t vec_size, size_t nP)
      : PreprocGate<R>(), mask_R(mask_R), tp_mask_R(tp_mask_R),
        permuted_masks(permuted_masks), tp_permuted_masks(tp_permuted_masks),
        vec_size(vec_size), nP(nP) {}
};


// Preprocessing for Rewire gate
// Rewires gates based on a position map (provided as a public permutation)
// Takes a position map vector and any number of payload vectors as input
// Outputs permuted payload wires based on the position map
// 
// Gate behavior (similar to lines 678-690 in compactEvaluate):
//   Apply public permutation to all payload vectors based on the position map
//   For each position i: if position_map[i] = idx_perm, then output[idx_perm] = payload[i]
//
// This gate requires:
//   - No preprocessing needed (permutation is public)
template <class R>
struct PreprocRewireGate : public PreprocGate<R> {
  size_t vec_size;       // Size of position map and each payload vector
  size_t num_payloads;   // Number of payload vectors
  
  PreprocRewireGate() = default;
  PreprocRewireGate(size_t vec_size, size_t num_payloads)
      : PreprocGate<R>(), vec_size(vec_size), num_payloads(num_payloads) {}
};


// Preprocessed data for the circuit.
template <class R>
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t<R>> gates;
  PreprocCircuit() = default;
};
};  // namespace graphdb
