#pragma once

#include <algorithm>
#include <array>
#include <boost/format.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "helpers.h"
#include "types.h"

namespace common::utils {

using wire_t = size_t;

enum GateType {
  kInp,
  kRec,
  kAdd,
  kMul,
  kSub,
  kEqz,
  kConstAdd,
  kConstMul,
  kShuffle,
  kPermAndSh,
  kPublicPerm,
  kCompact,
  kInvalid,
  NumGates
};

std::ostream& operator<<(std::ostream& os, GateType type);

// Gates represent primitive operations.
// All gates have one output.
struct Gate {
  GateType type{GateType::kInvalid};
  int owner;
  wire_t out;
  std::vector<wire_t> outs;
  std::vector<std::vector<wire_t>> multi_outs;

  Gate() = default;
  Gate(GateType type, wire_t out);
  Gate(GateType type, int owner, wire_t out, std::vector<wire_t> outs);
  Gate(GateType type, int owner, wire_t out, std::vector<std::vector<wire_t>> multi_outs);

  virtual ~Gate() = default;
};

// Represents a gate with fan-in 2.
struct FIn2Gate : public Gate {
  wire_t in1{0};
  wire_t in2{0};

  FIn2Gate() = default;
  FIn2Gate(GateType type, wire_t in1, wire_t in2, wire_t out);
};

struct FIn3Gate : public Gate {
  wire_t in1{0};
  wire_t in2{0};
  wire_t in3{0};

  FIn3Gate() = default;
  FIn3Gate(GateType type, wire_t in1, wire_t in2, wire_t in3, wire_t out);
};

struct FIn4Gate : public Gate {
  wire_t in1{0};
  wire_t in2{0};
  wire_t in3{0};
  wire_t in4{0};

  FIn4Gate() = default;
  FIn4Gate(GateType type, wire_t in1, wire_t in2, wire_t in3, wire_t in4, wire_t out);
};

// Represents a gate with fan-in 1.
struct FIn1Gate : public Gate {
  wire_t in{0};

  FIn1Gate() = default;
  FIn1Gate(GateType type, wire_t in, wire_t out);
};

// Represents a gate used to denote SIMD operations.
// These type is used to represent operations that take vectors of inputs but
// might not necessarily be SIMD e.g., dot product.
struct SIMDGate : public Gate {
  std::vector<wire_t> in1{0};
  std::vector<wire_t> in2{0};

  SIMDGate() = default;
  SIMDGate(GateType type, std::vector<wire_t> in1, std::vector<wire_t> in2, wire_t out);
};

// Represents a gate used to denote SIMD operations.
// These type is used to represent operations that take vectors of inputs and give vector of output but
// might not necessarily be SIMD e.g., shuffle, permute+share.
struct SIMDOGate : public Gate {
  std::vector<wire_t> in{0};
  std::vector<std::vector<int>> permutation{0};

  SIMDOGate() = default;
  SIMDOGate(GateType type, int owner, std::vector<wire_t> in, std::vector<wire_t> out, std::vector<std::vector<int>> permutation);
};

// Represents a gate used to denote SIMD operations.
// These type is used to represent operations that take vectors of inputs and give 2D vector of output but
// might not necessarily be SIMD e.g., amortized permute+share.
struct SIMDMOGate : public Gate {
  std::vector<wire_t> in{0};
  std::vector<std::vector<int>> permutation{0};

  SIMDMOGate() = default;
  SIMDMOGate(GateType type, int owner, std::vector<wire_t> in, std::vector<std::vector<wire_t>> multi_outs,
             std::vector<std::vector<int>> permutation);
};

// Represents gates where one input is a constant.
template <class R>
struct ConstOpGate : public Gate {
  wire_t in{0};
  R cval;

  ConstOpGate() = default;
  ConstOpGate(GateType type, wire_t in, R cval, wire_t out)
      : Gate(type, out), in(in), cval(std::move(cval)) {}
};

using gate_ptr_t = std::shared_ptr<Gate>;

// Gates ordered by multiplicative depth.
//
// Addition gates are not considered to increase the depth.
// Moreover, if gates_by_level[l][i]'s output is input to gates_by_level[l][j]
// then i < j.
struct LevelOrderedCircuit {
  size_t num_gates;
  size_t num_wires;
  std::array<uint64_t, GateType::NumGates> count;
  std::vector<wire_t> outputs;
  std::vector<std::vector<gate_ptr_t>> gates_by_level;

  friend std::ostream& operator<<(std::ostream& os, const LevelOrderedCircuit& circ);
};

// Represents an arithmetic circuit.
template <class R>
class Circuit {
  std::vector<wire_t> outputs_;
  std::vector<gate_ptr_t> gates_;
  size_t num_wires;

  bool isWireValid(wire_t wid) { return wid < num_wires; }

 public:
  Circuit() : num_wires(0) {}

  // Methods to manually build a circuit.
  wire_t newInputWire() {
    wire_t wid = num_wires;
    gates_.push_back(std::make_shared<Gate>(GateType::kInp, wid));
    num_wires += 1;
    return wid;
  }

  void setAsOutput(wire_t wid) {
    if (!isWireValid(wid)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    outputs_.push_back(wid);
  }

  // Function to add a gate with fan-in 2.
  wire_t addGate(GateType type, wire_t input1, wire_t input2) {
    if (type != GateType::kAdd && type != GateType::kMul &&
        type != GateType::kSub) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (!isWireValid(input1) || !isWireValid(input2)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<FIn2Gate>(type, input1, input2, output));
    num_wires += 1;

    return output;
  }

  // Function to add a gate with fan-in 3.
  // wire_t addGate(GateType type, wire_t input1, wire_t input2, wire_t input3) {
  //   if (type != GateType::kMul3) {
  //     throw std::invalid_argument("Invalid gate type.");
  //   }

  //   if (!isWireValid(input1) || !isWireValid(input2) || !isWireValid(input3)) {
  //     throw std::invalid_argument("Invalid wire ID.");
  //   }

  //   wire_t output = num_wires;
  //   gates_.push_back(std::make_shared<FIn3Gate>(type, input1, input2, input3, output));
  //   num_wires += 1;

  //   return output;
  // }

  // Function to add a gate with fan-in 4.
  // wire_t addGate(GateType type, wire_t input1, wire_t input2, 
  //                               wire_t input3, wire_t input4) {
  //   if (type != GateType::kMul4) {
  //     throw std::invalid_argument("Invalid gate type.");
  //   }

  //   if (!isWireValid(input1) || !isWireValid(input2) 
  //       || !isWireValid(input3) || !isWireValid(input4)) {
  //     throw std::invalid_argument("Invalid wire ID.");
  //   }

  //   wire_t output = num_wires;
  //   gates_.push_back(std::make_shared<FIn4Gate>(type, input1, input2, 
  //                                         input3, input4, output));
  //   num_wires += 1;

  //   return output;
  // }

  // Function to add a gate with one input from a wire and a second constant
  // input.
  wire_t addConstOpGate(GateType type, wire_t wid, R cval) {
    if (type != kConstAdd && type != kConstMul) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (!isWireValid(wid)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<ConstOpGate<R>>(type, wid, cval, output));
    num_wires += 1;

    return output;
  }

  // Function to add a single input gate.
  wire_t addGate(GateType type, wire_t input) {
    if (type != GateType::kEqz && type != GateType::kRec) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (!isWireValid(input)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<FIn1Gate>(type, input, output));
    num_wires += 1;

    return output;
  }

  // Function to add a multiple fan-in gate.
  // wire_t addGate(GateType type, const std::vector<wire_t>& input1,
  //                const std::vector<wire_t>& input2) {
  //   if (type != GateType::kDotprod && type != GateType::kTrdotp) {
  //     throw std::invalid_argument("Invalid gate type.");
  //   }

  //   if (input1.size() != input2.size()) {
  //     throw std::invalid_argument("Expected same length inputs.");
  //   }

  //   for (size_t i = 0; i < input1.size(); ++i) {
  //     if (!isWireValid(input1[i]) || !isWireValid(input2[i])) {
  //       throw std::invalid_argument("Invalid wire ID.");
  //     }
  //   }

  //   wire_t output = num_wires;
  //   gates_.push_back(std::make_shared<SIMDGate>(type, input1, input2, output));
  //   num_wires += 1;
  //   return output;
  // }

  // Function to add a multiple in + out gate.
  std::vector<wire_t> addMGate(GateType type, const std::vector<wire_t>& input, const std::vector<std::vector<int>> &permutation,
                               int owner = 0) {
    if (type != GateType::kShuffle && type != GateType::kPermAndSh) {
      throw std::invalid_argument("Invalid gate type.");
    }

    for (size_t i = 0; i < input.size(); i++) {
      if (!isWireValid(input[i])) {
        throw std::invalid_argument("Invalid wire ID.");
      }
    }

    if (permutation.size() == 0) {
      throw std::invalid_argument("No permutation passed.");
    }

    for (size_t i = 0; i < permutation.size(); ++i) {
      if (input.size() != permutation[i].size()) {
        throw std::invalid_argument("Permutation size mismatch.");
      }
    }

    std::vector<wire_t> output(input.size());
    for (int i = 0; i < input.size(); i++) {
      output[i] = i + num_wires;
    }
    gates_.push_back(std::make_shared<SIMDOGate>(type, owner, input, output, permutation));
    num_wires += input.size();
    return output;
  }

  std::vector<wire_t> addConstOpMGate(GateType type, const std::vector<wire_t>& input, const std::vector<int> &permutation) {
    if (type != GateType::kPublicPerm) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (input.size() != permutation.size()) {
      throw std::invalid_argument("Permutation size mismatch.");
    }

    for (size_t i = 0; i < input.size(); i++) {
      if (!isWireValid(input[i])) {
        throw std::invalid_argument("Invalid wire ID.");
      }
    }

    std::vector<std::vector<int>> permutation_wrapper(1);
    permutation_wrapper[0] = std::move(permutation);

    std::vector<wire_t> output(input.size());
    for (int i = 0; i < input.size(); i++) {
      output[i] = i + num_wires;
    }
    gates_.push_back(std::make_shared<SIMDOGate>(type, 0, input, output, permutation_wrapper));
    num_wires += input.size();
    return output;
  }

  // Function to add a multiple in + out gate.
  // std::vector<std::vector<wire_t>> addMOGate(GateType type, const std::vector<wire_t>& input, const std::vector<std::vector<int>> &permutation,
  //                                            int nP) {
  //   if (type != GateType::kAmortzdPnS) {
  //     throw std::invalid_argument("Invalid gate type.");
  //   }

  //   for (size_t i = 0; i < input.size(); i++) {
  //     if (!isWireValid(input[i])) {
  //       throw std::invalid_argument("Invalid wire ID.");
  //     }
  //   }

  //   if (permutation.size() == 0) {
  //     throw std::invalid_argument("No permutation passed.");
  //   }

  //   for (size_t i = 0; i < permutation.size(); ++i) {
  //     if (input.size() != permutation[i].size()) {
  //       throw std::invalid_argument("Permutation size mismatch.");
  //     }
  //   }

  //   std::vector<std::vector<wire_t>> output(nP, std::vector<wire_t>(input.size()));
  //   for (int pid = 0; pid < nP; ++pid) {
  //     for (int i = 0; i < input.size(); i++) {
  //       output[pid][i] = pid * nP + i + num_wires;
  //     }
  //   }
  //   gates_.push_back(std::make_shared<SIMDMOGate>(type, 0, input, output, permutation));
  //   num_wires += nP * input.size();
  //   return output;
  // }

  // Add a compaction gate that takes t (tags) and multiple payload vectors
  // Returns compacted versions of t and all payload vectors
  std::pair<std::vector<wire_t>, std::vector<std::vector<wire_t>>> addCompactGate(
      const std::vector<wire_t>& t_vector,
      const std::vector<std::vector<wire_t>>& p_vectors,
      const std::vector<std::vector<int>>& permutation) {
    
    size_t vec_size = t_vector.size();
    size_t num_payloads = p_vectors.size();
    
    if (num_payloads == 0) {
      throw std::invalid_argument("At least one payload vector is required.");
    }
    
    // Validate all payload vectors have same size as t_vector
    for (size_t p = 0; p < num_payloads; ++p) {
      if (p_vectors[p].size() != vec_size) {
        throw std::invalid_argument("All payload vectors must have the same size as t_vector.");
      }
    }
    
    // Validate input wires
    for (size_t i = 0; i < vec_size; i++) {
      if (!isWireValid(t_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in t_vector.");
      }
      for (size_t p = 0; p < num_payloads; ++p) {
        if (!isWireValid(p_vectors[p][i])) {
          throw std::invalid_argument("Invalid wire ID in payload vector.");
        }
      }
    }
    
    // Create output wires: vec_size for t_compacted, vec_size for each p_compacted
    std::vector<wire_t> t_compacted(vec_size);
    std::vector<std::vector<wire_t>> p_compacted(num_payloads, std::vector<wire_t>(vec_size));
    
    for (size_t i = 0; i < vec_size; i++) {
      t_compacted[i] = num_wires + i;
    }
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        p_compacted[p][i] = num_wires + vec_size * (p + 1) + i;
      }
    }
    
    // Create input vector [t0, t1, ..., tn, p1_0, p1_1, ..., p1_n, p2_0, ...]
    std::vector<wire_t> input((1 + num_payloads) * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      input[i] = t_vector[i];
    }
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        input[vec_size * (p + 1) + i] = p_vectors[p][i];
      }
    }
    
    // Create output vector for the gate
    std::vector<wire_t> output((1 + num_payloads) * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      output[i] = t_compacted[i];
    }
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        output[vec_size * (p + 1) + i] = p_compacted[p][i];
      }
    }
    
    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kCompact, 0, input, output, permutation));
    num_wires += (1 + num_payloads) * vec_size;
    
    return {t_compacted, p_compacted};
  }





  
  // Level ordered gates are helpful for evaluation.
  [[nodiscard]] LevelOrderedCircuit orderGatesByLevel() const {
    LevelOrderedCircuit res;
    res.outputs = outputs_;
    res.num_gates = gates_.size();
    res.num_wires = num_wires;

    // Map from output wire id to multiplicative depth/level.
    // Input gates have a depth of 0.
    std::vector<size_t> gate_level(num_wires, 0);
    size_t depth = 0;

    // This assumes that if gates_[i]'s output is input to gates_[j] then
    // i < j.
    for (const auto& gate : gates_) {
      switch (gate->type) {
        case GateType::kRec: {
          const auto* g = static_cast<FIn1Gate*>(gate.get());
          gate_level[g->out] = gate_level[g->in] + 1;
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }
        case GateType::kAdd:
        case GateType::kSub: {
          const auto* g = static_cast<FIn2Gate*>(gate.get());
          gate_level[g->out] = std::max(gate_level[g->in1], gate_level[g->in2]);
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kMul: {
          const auto* g = static_cast<FIn2Gate*>(gate.get());
          gate_level[g->out] = std::max(gate_level[g->in1], gate_level[g->in2]) + 1;
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kConstAdd:
        case GateType::kConstMul: {
          const auto* g = static_cast<ConstOpGate<R>*>(gate.get());
          gate_level[g->out] = gate_level[g->in];
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kEqz: {
          const auto* g = static_cast<FIn1Gate*>(gate.get());
          gate_level[g->out] = gate_level[g->in] + 1;
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kShuffle:
        case GateType::kPermAndSh: {
          const auto* g = static_cast<SIMDOGate*>(gate.get());
          size_t gate_depth = 0;
          for (size_t i = 0; i < g->in.size(); i++) {
            gate_depth = std::max({gate_level[g->in[i]], gate_depth});
          }
          for (int i = 0; i < g->outs.size(); i++) {
            gate_level[g->outs[i]] = gate_depth + 1;
          }
          depth = std::max(depth, gate_level[gate->outs[0]]);
          break;
        }

        case GateType::kPublicPerm: {
          const auto* g = static_cast<SIMDOGate*>(gate.get());
          size_t gate_depth = 0;
          for (size_t i = 0; i < g->in.size(); i++) {
            gate_depth = std::max({gate_level[g->in[i]], gate_depth});
          }
          for (int i = 0; i < g->outs.size(); i++) {
            gate_level[g->outs[i]] = gate_depth;
          }
          depth = std::max(depth, gate_level[gate->outs[0]]);
          break;
        }

        case GateType::kCompact: {
          const auto* g = static_cast<SIMDOGate*>(gate.get());
          size_t gate_depth = 0;
          for (size_t i = 0; i < g->in.size(); i++) {
            gate_depth = std::max({gate_level[g->in[i]], gate_depth});
          }
          for (int i = 0; i < g->outs.size(); i++) {
            gate_level[g->outs[i]] = gate_depth + 1;
          }
          depth = std::max(depth, gate_level[gate->outs[0]]);
          break;
        }

        default:
          break;
      }
    }

    std::fill(res.count.begin(), res.count.end(), 0);

    std::vector<std::vector<gate_ptr_t>> gates_by_level(depth + 1);
    for (const auto& gate : gates_) {
      res.count[gate->type]++;
      if (gate->type == GateType::kShuffle || gate->type == GateType::kPermAndSh || gate->type == GateType::kPublicPerm || gate->type == GateType::kCompact) {
        gates_by_level[gate_level[gate->outs[0]]].push_back(gate);
      } else {
        gates_by_level[gate_level[gate->out]].push_back(gate);
      } 
    }

    res.gates_by_level = std::move(gates_by_level);

    return res;
  }

  // Evaluate circuit on plaintext inputs.
  [[nodiscard]] std::vector<R> evaluate(const std::unordered_map<wire_t, R>& inputs) const {
    auto level_circ = orderGatesByLevel();
    std::vector<R> wires(level_circ.num_gates);

    auto num_inp_gates = level_circ.count[GateType::kInp];
    if (inputs.size() != num_inp_gates) {
      throw std::invalid_argument(boost::str(
          boost::format("Expected %1% inputs but received %2% inputs.") %
          num_inp_gates % inputs.size()));
    }

    for (const auto& level : level_circ.gates_by_level) {
      for (const auto& gate : level) {
        switch (gate->type) {
          case GateType::kInp: {
            wires[gate->out] = inputs.at(gate->out);
            break;
          }

          case GateType::kMul: {
            auto* g = static_cast<FIn2Gate*>(gate.get());
            wires[g->out] = wires[g->in1] * wires[g->in2];
            break;
          }

          case GateType::kAdd: {
            auto* g = static_cast<FIn2Gate*>(gate.get());
            wires[g->out] = wires[g->in1] + wires[g->in2];
            break;
          }

          case GateType::kSub: {
            auto* g = static_cast<FIn2Gate*>(gate.get());
            wires[g->out] = wires[g->in1] - wires[g->in2];
            break;
          }

          case GateType::kConstAdd: {
            auto* g = static_cast<ConstOpGate<R>*>(gate.get());
            wires[g->out] = wires[g->in] + g->cval;
            break;
          }

          case GateType::kConstMul: {
            auto* g = static_cast<ConstOpGate<R>*>(gate.get());
            wires[g->out] = wires[g->in] * g->cval;
            break;
          }

          case GateType::kEqz: {
            auto* g = static_cast<FIn1Gate*>(gate.get());
            if (wires[g->in] == 0) {
              wires[g->out] = 1;
            }
            else {
              wires[g->out] = 0;
            }
            break;
          }

          default: {
            throw std::runtime_error("Invalid gate type.");
          }
        }
      }
    }

    std::vector<R> outputs;
    for (auto i : level_circ.outputs) {
      outputs.push_back(wires[i]);
    }

    return outputs;
  }

};
};  // namespace common::utils
