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
  // Random share for input masking.
  AuthAddShare<R> share_r;
  PreprocInput() = default;
  PreprocInput(int pid, const AuthAddShare<R>& share_r) 
      : PreprocGate<R>(), pid(pid), share_r(share_r) {}
  PreprocInput(const PreprocInput<R>& pregate) 
      : PreprocGate<R>(), pid(pregate.pid), share_r(pregate.share_r) {}
};

template <class R>
struct PreprocRecGate : public PreprocGate<R> {
  bool Pking = false;
  PreprocRecGate() = default;
  PreprocRecGate(bool Pking)
    : PreprocGate<R>(), Pking(Pking) {}
};

// Preprocessed data for the circuit.
template <class R>
struct PreprocCircuit {
  std::unordered_map<wire_t, preprocg_ptr_t<R>> gates;
  PreprocCircuit() = default;
};
}; 