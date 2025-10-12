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
struct PreprocShuffleGate : public PreprocGate<R> {
  std::vector<AddShare<R>> a; // Randomly sampled vector
  std::vector<TPShare<R>> tp_a; // Randomly sampled vector
  std::vector<AddShare<R>> b; // Randomly sampled vector
  std::vector<TPShare<R>> tp_b; // Randomly sampled vector
  std::vector<AddShare<R>> c; // Randomly sampled vector
  std::vector<TPShare<R>> tp_c; // Randomly sampled vector
  std::vector<AddShare<R>> delta; // Delta vector only held by the last party. Dummy values for the other parties
  std::vector<int> pi; // Randomly sampled permutation using HP
  std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutations of all parties using HP
  std::vector<int> pi_common; // Common random permutation held by all parties except HP. HP holds dummy values
  PreprocShuffleGate() = default;
  PreprocShuffleGate(const std::vector<AddShare<R>>& a, const std::vector<TPShare<R>>& tp_a,
                     const std::vector<AddShare<R>>& b, const std::vector<TPShare<R>>& tp_b,
                     const std::vector<AddShare<R>>& c, const std::vector<TPShare<R>>& tp_c,
                     const std::vector<AddShare<R>>& delta, const std::vector<int>& pi, const std::vector<std::vector<int>>& tp_pi_all,
                     const std::vector<int>& pi_common)
      : PreprocGate<R>(), a(a), tp_a(tp_a), b(b), tp_b(tp_b), c(c), tp_c(tp_c), delta(delta),
        pi(pi), tp_pi_all(tp_pi_all), pi_common(pi_common) {}
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

// Preprocessed data for the circuit.
template <class R>
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t<R>> gates;
  PreprocCircuit() = default;
};
};  // namespace graphdb
