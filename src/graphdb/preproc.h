#pragma once

#include "../utils/circuit.h"
#include "sharing.h"
#include "../utils/types.h"
#include <unordered_map>

using namespace common::utils;

namespace graphdb {
// Preprocessed data for a gate.
struct PreprocGate {
  PreprocGate() = default;
  virtual ~PreprocGate() = default;
};

using preprocg_ptr_t = std::unique_ptr<PreprocGate>;


struct PreprocInput : public PreprocGate {
  // ID of party providing input on wire.
  int pid{};
  // Random share for input masking.
  AuthAddShare share_r;
  Field r;
  PreprocInput() = default;
  PreprocInput(const AuthAddShare& share_r) 
      : PreprocGate(), pid(0), share_r(share_r), r(Field(0)) {}
  PreprocInput(int pid, const AuthAddShare& share_r, const Field& r) 
      : PreprocGate(), pid(pid), share_r(share_r), r(r) {}
};


struct PreprocMultGate : public PreprocGate {
  // Secret shared product of inputs masks.
  AuthAddShare triple_a; // Holds one beaver triple share of a random value a
  AuthAddShare triple_b; // Holds one beaver triple share of a random value b
  AuthAddShare triple_c; // Holds one beaver triple share of c=a*b

  PreprocMultGate() = default;
  PreprocMultGate(const AuthAddShare& triple_a,
                  const AuthAddShare& triple_b,
                  const AuthAddShare& triple_c)
      : PreprocGate(), triple_a(triple_a),
        triple_b(triple_b),
        triple_c(triple_c) {}
};


struct PreprocRecGate : public PreprocGate {
  bool Pking = false;
  PreprocRecGate() = default;
  PreprocRecGate(bool Pking)
    : PreprocGate(), Pking(Pking) {}
};

struct PreprocEqzGate : public PreprocGate {
  AuthAddShare share_r1;
  AuthAddShare share_r2;
  std::vector<AuthAddShare> share_r1_bits;
  std::vector<AuthAddShare> share_r2_bits;
  PreprocEqzGate() = default;
  PreprocEqzGate(const AuthAddShare& share_r1,
                 const AuthAddShare& share_r2,
                 const std::vector<AuthAddShare>& share_r1_bits,
                 const std::vector<AuthAddShare>& share_r2_bits)
      : PreprocGate(), share_r1(share_r1), share_r2(share_r2),
        share_r1_bits(share_r1_bits), share_r2_bits(share_r2_bits) {}
};


struct PreprocPermAndShGate : public PreprocGate {
  // Mask R and its shares
  std::vector<AuthAddShare> mask_R;      // This party's additive shares of R
  std::vector<AuthAddShare> mask_R_tag;      // This party's additive shares of R for tag


  // Permuted mask π_owner(R) (only one permutation per gate - the owner's)
  std::vector<AuthAddShare> permuted_mask;      // This party's additive shares of π_owner(R)
  std::vector<AuthAddShare> permuted_mask_tag;      // This party's additive shares of π_owner(R_tag)

  size_t vec_size; // Size of input vector
  int owner;       // Owner party id for this permute-and-share gate

  PreprocPermAndShGate() = default;
  PreprocPermAndShGate(const std::vector<AuthAddShare>& mask_R,
                       const std::vector<AuthAddShare>& permuted_mask,
                       const std::vector<AuthAddShare>& mask_R_tag,
                       const std::vector<AuthAddShare>& permuted_mask_tag,
                       size_t vec_size, int owner)
      : PreprocGate(), mask_R(mask_R), permuted_mask(permuted_mask),  mask_R_tag(mask_R_tag), permuted_mask_tag(permuted_mask_tag), vec_size(vec_size), owner(owner) {}
};


struct PreprocRewireGate : public PreprocGate {
  size_t vec_size;       // Size of position map and each payload vector
  size_t num_payloads;   // Number of payload vectors
  
  PreprocRewireGate() = default;
  PreprocRewireGate(size_t vec_size, size_t num_payloads)
      : PreprocGate(), vec_size(vec_size), num_payloads(num_payloads) {}
};

// Preprocessed data for the circuit.
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t> gates;
  PreprocCircuit() = default;
};

}; 