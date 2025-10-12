#pragma once

#include "../utils/circuit.h"
#include "sharing.h"
#include "../utils/types.h"
#include <unordered_map>

using namespace common::utils;

namespace emgraph {
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
struct PreprocMult3Gate : public PreprocGate<R> {
  // Secret shared product of inputs masks.
  AddShare<R> share_a; // Holds one share of a random value a
  TPShare<R> tp_share_a; // Holds all the shares of a random value a
  AddShare<R> share_b; // Holds one of a random value b
  TPShare<R> tp_share_b; // Holds all the shares of a random value b
  AddShare<R> share_c; // Holds one share of a random value c
  TPShare<R> tp_share_c; // Holds all the shares of a random value c
  AddShare<R> share_ab; // Holds one share of a*b
  TPShare<R> tp_share_ab; // Holds all the shares of a*b
  AddShare<R> share_bc; // Holds one share of b*c
  TPShare<R> tp_share_bc; // Holds all the shares of b*c
  AddShare<R> share_ca; // Holds one share of c*a
  TPShare<R> tp_share_ca; // Holds all the shares of c*a
  AddShare<R> share_abc; // Holds one share of a*b*c
  TPShare<R> tp_share_abc; // Holds all the shares of a*b*c
  PreprocMult3Gate() = default;
  PreprocMult3Gate(const AddShare<R>& share_a, const TPShare<R>& tp_share_a,
                   const AddShare<R>& share_b, const TPShare<R>& tp_share_b,
                   const AddShare<R>& share_c, const TPShare<R>& tp_share_c,
                   const AddShare<R>& share_ab, const TPShare<R>& tp_share_ab,
                   const AddShare<R>& share_bc, const TPShare<R>& tp_share_bc,
                   const AddShare<R>& share_ca, const TPShare<R>& tp_share_ca,
                   const AddShare<R>& share_abc, const TPShare<R>& tp_share_abc)
      : PreprocGate<R>(), share_a(share_a), tp_share_a(tp_share_a),
        share_b(share_b), tp_share_b(tp_share_b),
        share_c(share_c), tp_share_c(tp_share_c),
        share_ab(share_ab), tp_share_ab(tp_share_ab),
        share_bc(share_bc), tp_share_bc(tp_share_bc),
        share_ca(share_ca), tp_share_ca(tp_share_ca),
        share_abc(share_abc), tp_share_abc(tp_share_abc) {}
};

template <class R>
struct PreprocMult4Gate : public PreprocGate<R> {
  // Secret shared product of inputs masks.
  AddShare<R> share_a; // Holds one share of a random value a
  TPShare<R> tp_share_a; // Holds all the shares of a random value a
  AddShare<R> share_b; // Holds one of a random value b
  TPShare<R> tp_share_b; // Holds all the shares of a random value b
  AddShare<R> share_c; // Holds one share of a random value c
  TPShare<R> tp_share_c; // Holds all the shares of a random value c
  AddShare<R> share_d; // Holds one share of a random value d
  TPShare<R> tp_share_d; // Holds all the shares of a random value d
  AddShare<R> share_ab; // Holds one share of a*b
  TPShare<R> tp_share_ab; // Holds all the shares of a*b
  AddShare<R> share_ac; // Holds one share of a*c
  TPShare<R> tp_share_ac; // Holds all the shares of a*c
  AddShare<R> share_ad; // Holds one share of a*d
  TPShare<R> tp_share_ad; // Holds all the shares of a*d
  AddShare<R> share_bc; // Holds one share of b*c
  TPShare<R> tp_share_bc; // Holds all the shares of b*c
  AddShare<R> share_bd; // Holds one share of b*d
  TPShare<R> tp_share_bd; // Holds all the shares of b*d
  AddShare<R> share_cd; // Holds one share of c*d
  TPShare<R> tp_share_cd; // Holds all the shares of c*d
  AddShare<R> share_abc; // Holds one share of a*b*c
  TPShare<R> tp_share_abc; // Holds all the shares of a*b*c
  AddShare<R> share_abd; // Holds one share of a*b*d
  TPShare<R> tp_share_abd; // Holds all the shares of a*b*d
  AddShare<R> share_acd; // Holds one share of a*c*d
  TPShare<R> tp_share_acd; // Holds all the shares of a*c*d
  AddShare<R> share_bcd; // Holds one share of b*c*d
  TPShare<R> tp_share_bcd; // Holds all the shares of b*c*d
  AddShare<R> share_abcd; // Holds one share of a*b*c*d
  TPShare<R> tp_share_abcd; // Holds all the shares of a*b*c*d
  PreprocMult4Gate() = default;
  PreprocMult4Gate(const AddShare<R>& share_a, const TPShare<R>& tp_share_a,
                   const AddShare<R>& share_b, const TPShare<R>& tp_share_b,
                   const AddShare<R>& share_c, const TPShare<R>& tp_share_c,
                   const AddShare<R>& share_d, const TPShare<R>& tp_share_d,
                   const AddShare<R>& share_ab, const TPShare<R>& tp_share_ab,
                   const AddShare<R>& share_ac, const TPShare<R>& tp_share_ac,
                   const AddShare<R>& share_ad, const TPShare<R>& tp_share_ad,
                   const AddShare<R>& share_bc, const TPShare<R>& tp_share_bc,
                   const AddShare<R>& share_bd, const TPShare<R>& tp_share_bd,
                   const AddShare<R>& share_cd, const TPShare<R>& tp_share_cd,
                   const AddShare<R>& share_abc, const TPShare<R>& tp_share_abc,
                   const AddShare<R>& share_abd, const TPShare<R>& tp_share_abd,
                   const AddShare<R>& share_acd, const TPShare<R>& tp_share_acd,
                   const AddShare<R>& share_bcd, const TPShare<R>& tp_share_bcd,
                   const AddShare<R>& share_abcd, const TPShare<R>& tp_share_abcd)
      : PreprocGate<R>(), share_a(share_a), tp_share_a(tp_share_a),
        share_b(share_b), tp_share_b(tp_share_b),
        share_c(share_c), tp_share_c(tp_share_c),
        share_d(share_d), tp_share_d(tp_share_d),
        share_ab(share_ab), tp_share_ab(tp_share_ab),
        share_ac(share_ac), tp_share_ac(tp_share_ac),
        share_ad(share_ad), tp_share_ad(tp_share_ad),
        share_bc(share_bc), tp_share_bc(tp_share_bc),
        share_bd(share_bd), tp_share_bd(tp_share_bd),
        share_cd(share_cd), tp_share_cd(tp_share_cd),
        share_abc(share_abc), tp_share_abc(tp_share_abc),
        share_abd(share_abd), tp_share_abd(tp_share_abd),
        share_acd(share_acd), tp_share_acd(tp_share_acd),
        share_bcd(share_bcd), tp_share_bcd(tp_share_bcd),
        share_abcd(share_abcd), tp_share_abcd(tp_share_abcd) {}
};

template <class R>
struct PreprocDotpGate : public PreprocGate<R> {
  std::vector<AddShare<R>> triple_a_vec{};
  std::vector<TPShare<R>> tp_triple_a_vec{};
  std::vector<AddShare<R>> triple_b_vec{};
  std::vector<TPShare<R>> tp_triple_b_vec{};
  std::vector<AddShare<R>> triple_c_vec{};
  std::vector<TPShare<R>> tp_triple_c_vec{};
  PreprocDotpGate() = default;
  PreprocDotpGate(const std::vector<AddShare<R>>& triple_a_vec, const std::vector<TPShare<R>>& tp_triple_a_vec,
                  const std::vector<AddShare<R>>& triple_b_vec, const std::vector<TPShare<R>>& tp_triple_b_vec,
                  const std::vector<AddShare<R>>& triple_c_vec, const std::vector<TPShare<R>>& tp_triple_c_vec)
      : PreprocGate<R>(), triple_a_vec(triple_a_vec), tp_triple_a_vec(tp_triple_a_vec),
        triple_b_vec(triple_b_vec), tp_triple_b_vec(tp_triple_b_vec),
        triple_c_vec(triple_c_vec), tp_triple_c_vec(tp_triple_c_vec) {}
};

template <class R>
struct PreprocEqzGate : public PreprocGate<R> {
  AddShare<R> share_r;
  TPShare<R> tp_share_r;
  std::vector<AddShare<BoolRing>> share_r_bits;
  std::vector<TPShare<BoolRing>> tp_share_r_bits;
  std::vector<preprocg_ptr_t<BoolRing>> multk_gates;
  PreprocEqzGate() = default;
  PreprocEqzGate(const AddShare<R> &share_r, const TPShare<R> &tp_share_r,
                 const std::vector<AddShare<BoolRing>> &share_r_bits, const std::vector<TPShare<BoolRing>> &tp_share_r_bits,
                 std::vector<preprocg_ptr_t<BoolRing>> multk_gates)
    : PreprocGate<R>(), share_r(share_r), tp_share_r(tp_share_r), share_r_bits(share_r_bits), tp_share_r_bits(tp_share_r_bits),
      multk_gates(std::move(multk_gates)) {}
};

template <class R>
struct PreprocLtzGate : public PreprocGate<R> {
  AddShare<R> share_r;
  TPShare<R> tp_share_r;
  std::vector<AddShare<BoolRing>> share_r_bits;
  std::vector<TPShare<BoolRing>> tp_share_r_bits;
  std::vector<preprocg_ptr_t<BoolRing>> PrefixOR_gates;
  PreprocLtzGate() = default;
  PreprocLtzGate(const AddShare<R> &share_r, const TPShare<R> &tp_share_r,
                 const std::vector<AddShare<BoolRing>> &share_r_bits, const std::vector<TPShare<BoolRing>> &tp_share_r_bits,
                 std::vector<preprocg_ptr_t<BoolRing>> PrefixOR_gates)
    : PreprocGate<R>(), share_r(share_r), tp_share_r(tp_share_r), share_r_bits(share_r_bits), tp_share_r_bits(tp_share_r_bits),
      PrefixOR_gates(std::move(PrefixOR_gates)) {}
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

template <class R>
struct PreprocAmortzdPnSGate : public PreprocGate<R> {
  std::vector<AddShare<R>> a; // Randomly sampled vector
  std::vector<TPShare<R>> tp_a; // Randomly sampled vector
  std::vector<AddShare<R>> b; // Randomly sampled vector
  std::vector<TPShare<R>> tp_b; // Randomly sampled vector
  std::vector<AddShare<R>> delta; // Delta vector only held by the last party. Dummy values for the other parties
  std::vector<int> pi; // Randomly sampled permutation using HP
  std::vector<std::vector<int>> tp_pi_all; // Randomly sampled permutations of all parties using HP
  std::vector<int> pi_common; // Common random permutation held by all parties except HP. HP holds dummy values
  PreprocAmortzdPnSGate() = default;
  PreprocAmortzdPnSGate(const std::vector<AddShare<R>>& a, const std::vector<TPShare<R>>& tp_a,
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
};  // namespace emgraph
