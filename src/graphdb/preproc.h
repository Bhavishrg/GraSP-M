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
  std::vector<AddShare<R>> a; // Randomly sampled vector
  std::vector<TPShare<R>> tp_a; // Randomly sampled vector
  std::vector<AddShare<R>> b; // Randomly sampled vector
  std::vector<TPShare<R>> tp_b; // Randomly sampled vector
  std::vector<AddShare<R>> delta; // Delta vector only held by the owner party. Dummy values for the other parties
  std::vector<int> pi; // Randomly sampled permutation using HP
  std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutations of all parties using HP
  std::vector<int> pi_common; // Common random permutation held by all parties except HP. HP holds dummy values
  PreprocPermAndShGate() = default;
  PreprocPermAndShGate(const std::vector<AddShare<R>>& a, const std::vector<TPShare<R>>& tp_a,
                       const std::vector<AddShare<R>>& b, const std::vector<TPShare<R>>& tp_b,
                       const std::vector<AddShare<R>>& delta, const std::vector<int>& pi, const std::vector<std::vector<int>>& tp_pi_all,
                       const std::vector<int>& pi_common)
      : PreprocGate<R>(), a(a), tp_a(tp_a), b(b), tp_b(tp_b), delta(delta), pi(pi), tp_pi_all(tp_pi_all), pi_common(pi_common) {}
};

template <class R>
struct PreprocCompactGate : public PreprocGate<R> {
  // Preprocessing for shuffle operations
  std::vector<AddShare<R>> shuffle_a;
  std::vector<TPShare<R>> shuffle_tp_a;
  std::vector<AddShare<R>> shuffle_b;
  std::vector<TPShare<R>> shuffle_tp_b;
  std::vector<AddShare<R>> shuffle_c;
  std::vector<TPShare<R>> shuffle_tp_c;
  std::vector<Ring> shuffle_delta;
  std::vector<int> shuffle_pi;
  std::vector<std::vector<int>> shuffle_tp_pi_all;
  
  // Preprocessing for multiplications
  std::vector<AddShare<R>> mult_triple_a;
  std::vector<TPShare<R>> mult_tp_triple_a;
  std::vector<AddShare<R>> mult_triple_b;
  std::vector<TPShare<R>> mult_tp_triple_b;
  std::vector<AddShare<R>> mult_triple_c;
  std::vector<TPShare<R>> mult_tp_triple_c;
  
  PreprocCompactGate() = default;
  PreprocCompactGate(const std::vector<AddShare<R>>& shuffle_a, const std::vector<TPShare<R>>& shuffle_tp_a,
                     const std::vector<AddShare<R>>& shuffle_b, const std::vector<TPShare<R>>& shuffle_tp_b,
                     const std::vector<AddShare<R>>& shuffle_c, const std::vector<TPShare<R>>& shuffle_tp_c,
                     const std::vector<R>& shuffle_delta, const std::vector<int>& shuffle_pi,
                     const std::vector<std::vector<int>>& shuffle_tp_pi_all,
                     const std::vector<AddShare<R>>& mult_triple_a, const std::vector<TPShare<R>>& mult_tp_triple_a,
                     const std::vector<AddShare<R>>& mult_triple_b, const std::vector<TPShare<R>>& mult_tp_triple_b,
                     const std::vector<AddShare<R>>& mult_triple_c, const std::vector<TPShare<R>>& mult_tp_triple_c)
      : PreprocGate<R>(), shuffle_a(shuffle_a), shuffle_tp_a(shuffle_tp_a), shuffle_b(shuffle_b), shuffle_tp_b(shuffle_tp_b),
        shuffle_c(shuffle_c), shuffle_tp_c(shuffle_tp_c), shuffle_delta(shuffle_delta), shuffle_pi(shuffle_pi),
        shuffle_tp_pi_all(shuffle_tp_pi_all), mult_triple_a(mult_triple_a), mult_tp_triple_a(mult_tp_triple_a),
        mult_triple_b(mult_triple_b), mult_tp_triple_b(mult_tp_triple_b), mult_triple_c(mult_triple_c),
        mult_tp_triple_c(mult_tp_triple_c) {}
};

// Preprocessing for Group-wise Index gate
// Uses two compact operations and multiplication triples for secure multiplications
template <class R>
struct PreprocGroupwiseIndexGate : public PreprocGate<R> {
  // First compaction preprocessing (for key vector)
  std::vector<AddShare<R>> shuffle_a;
  std::vector<TPShare<R>> shuffle_tp_a;
  std::vector<AddShare<R>> shuffle_b;
  std::vector<TPShare<R>> shuffle_tp_b;
  std::vector<AddShare<R>> shuffle_c;
  std::vector<TPShare<R>> shuffle_tp_c;
  std::vector<Ring> shuffle_delta;
  std::vector<int> shuffle_pi;
  std::vector<std::vector<int>> shuffle_tp_pi_all;
  std::vector<AddShare<R>> mult_triple_a;
  std::vector<TPShare<R>> mult_tp_triple_a;
  std::vector<AddShare<R>> mult_triple_b;
  std::vector<TPShare<R>> mult_tp_triple_b;
  std::vector<AddShare<R>> mult_triple_c;
  std::vector<TPShare<R>> mult_tp_triple_c;
  
  // Reverse compaction preprocessing (shuffle only, no multiplication needed)
  std::vector<AddShare<R>> revcompact_shuffle_a;
  std::vector<TPShare<R>> revcompact_shuffle_tp_a;
  std::vector<AddShare<R>> revcompact_shuffle_b;
  std::vector<TPShare<R>> revcompact_shuffle_tp_b;
  std::vector<AddShare<R>> revcompact_shuffle_c;
  std::vector<TPShare<R>> revcompact_shuffle_tp_c;
  std::vector<Ring> revcompact_shuffle_delta;
  std::vector<int> revcompact_shuffle_pi;
  std::vector<std::vector<int>> revcompact_shuffle_tp_pi_all;
  
  // Multiplication triples for key_c * key_compacted
  std::vector<AddShare<R>> keymult_triple_a;
  std::vector<TPShare<R>> keymult_tp_triple_a;
  std::vector<AddShare<R>> keymult_triple_b;
  std::vector<TPShare<R>> keymult_tp_triple_b;
  std::vector<AddShare<R>> keymult_triple_c;
  std::vector<TPShare<R>> keymult_tp_triple_c;
  
  PreprocGroupwiseIndexGate() = default;
};

// Preprocessing for Group-wise Propagate gate
// Similar structure to GroupwiseIndex but with different multiplication semantics
template <class R>
struct PreprocGroupwisePropagateGate : public PreprocGate<R> {
  // First compaction preprocessing (for T1 based on key)
  std::vector<AddShare<R>> t1_shuffle_a;
  std::vector<TPShare<R>> t1_shuffle_tp_a;
  std::vector<AddShare<R>> t1_shuffle_b;
  std::vector<TPShare<R>> t1_shuffle_tp_b;
  std::vector<AddShare<R>> t1_shuffle_c;
  std::vector<TPShare<R>> t1_shuffle_tp_c;
  std::vector<Ring> t1_shuffle_delta;
  std::vector<int> t1_shuffle_pi;
  std::vector<std::vector<int>> t1_shuffle_tp_pi_all;
  std::vector<AddShare<R>> t1_mult_triple_a;
  std::vector<TPShare<R>> t1_mult_tp_triple_a;
  std::vector<AddShare<R>> t1_mult_triple_b;
  std::vector<TPShare<R>> t1_mult_tp_triple_b;
  std::vector<AddShare<R>> t1_mult_triple_c;
  std::vector<TPShare<R>> t1_mult_tp_triple_c;
  
  // Second compaction preprocessing (for T2 based on key)
  std::vector<AddShare<R>> t2_shuffle_a;
  std::vector<TPShare<R>> t2_shuffle_tp_a;
  std::vector<AddShare<R>> t2_shuffle_b;
  std::vector<TPShare<R>> t2_shuffle_tp_b;
  std::vector<AddShare<R>> t2_shuffle_c;
  std::vector<TPShare<R>> t2_shuffle_tp_c;
  std::vector<Ring> t2_shuffle_delta;
  std::vector<int> t2_shuffle_pi;
  std::vector<std::vector<int>> t2_shuffle_tp_pi_all;
  std::vector<AddShare<R>> t2_mult_triple_a;
  std::vector<TPShare<R>> t2_mult_tp_triple_a;
  std::vector<AddShare<R>> t2_mult_triple_b;
  std::vector<TPShare<R>> t2_mult_tp_triple_b;
  std::vector<AddShare<R>> t2_mult_triple_c;
  std::vector<TPShare<R>> t2_mult_tp_triple_c;
  
  // Multiplication triples for difference * key computation in Step 3
  std::vector<AddShare<R>> diff_mult_triple_a;
  std::vector<TPShare<R>> diff_mult_tp_triple_a;
  std::vector<AddShare<R>> diff_mult_triple_b;
  std::vector<TPShare<R>> diff_mult_tp_triple_b;
  std::vector<AddShare<R>> diff_mult_triple_c;
  std::vector<TPShare<R>> diff_mult_tp_triple_c;
  
  // Reverse compaction preprocessing (for Step 4)
  std::vector<AddShare<R>> revcompact_shuffle_a;
  std::vector<TPShare<R>> revcompact_shuffle_tp_a;
  std::vector<AddShare<R>> revcompact_shuffle_b;
  std::vector<TPShare<R>> revcompact_shuffle_tp_b;
  std::vector<AddShare<R>> revcompact_shuffle_c;
  std::vector<TPShare<R>> revcompact_shuffle_tp_c;
  std::vector<Ring> revcompact_shuffle_delta;
  std::vector<int> revcompact_shuffle_pi;
  std::vector<std::vector<int>> revcompact_shuffle_tp_pi_all;
  
  PreprocGroupwisePropagateGate() = default;
};

// Preprocessing for Sort gate
// Contains 32 compaction operations (one for each bit from MSB to LSB)
// Plus final shuffle for hiding the permutation
template <class R>
struct PreprocSortGate : public PreprocGate<R> {
  // Array of 32 compaction preprocessing structures (one per bit)
  // Index 0 corresponds to MSB (bit 31), index 31 corresponds to LSB (bit 0)
  std::vector<std::vector<AddShare<R>>> shuffle_a;  // [32][vec_size]
  std::vector<std::vector<TPShare<R>>> shuffle_tp_a;
  std::vector<std::vector<AddShare<R>>> shuffle_b;
  std::vector<std::vector<TPShare<R>>> shuffle_tp_b;
  std::vector<std::vector<AddShare<R>>> shuffle_c;
  std::vector<std::vector<TPShare<R>>> shuffle_tp_c;
  std::vector<std::vector<Ring>> shuffle_delta;  // [32][vec_size]
  std::vector<std::vector<int>> shuffle_pi;  // [32][vec_size]
  std::vector<std::vector<std::vector<int>>> shuffle_tp_pi_all;  // [32][nP][vec_size]
  
  // Multiplication triples for each of the 32 compact operations
  std::vector<std::vector<AddShare<R>>> mult_triple_a;  // [32][vec_size]
  std::vector<std::vector<TPShare<R>>> mult_tp_triple_a;
  std::vector<std::vector<AddShare<R>>> mult_triple_b;
  std::vector<std::vector<TPShare<R>>> mult_tp_triple_b;
  std::vector<std::vector<AddShare<R>>> mult_triple_c;
  std::vector<std::vector<TPShare<R>>> mult_tp_triple_c;
  
  // Final shuffle preprocessing for hiding the permutation before reconstruction
  std::vector<AddShare<R>> final_shuffle_a;
  std::vector<TPShare<R>> final_shuffle_tp_a;
  std::vector<AddShare<R>> final_shuffle_b;
  std::vector<TPShare<R>> final_shuffle_tp_b;
  std::vector<AddShare<R>> final_shuffle_c;
  std::vector<TPShare<R>> final_shuffle_tp_c;
  std::vector<Ring> final_shuffle_delta;
  std::vector<int> final_shuffle_pi;
  std::vector<std::vector<int>> final_shuffle_tp_pi_all;
  
  PreprocSortGate() = default;
};

// Preprocessed data for the circuit.
template <class R>
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t<R>> gates;
  PreprocCircuit() = default;
};
};  // namespace graphdb
