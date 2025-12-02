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

// Preprocessed data for the circuit.
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t> gates;
  PreprocCircuit() = default;
};

}; 