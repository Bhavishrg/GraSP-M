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
  kAdd,
  kMul,
  kMul3,
  kMul4,
  kSub,
  kConstAdd,
  kConstMul,
  kRelu,
  kMsb,
  kEqz,
  kLtz,
  kDotprod,
  kTrdotp,
  kShuffle,
  kPermAndSh,
  kAmortzdPnS,
  kPublicPerm,
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
  wire_t addGate(GateType type, wire_t input1, wire_t input2, wire_t input3) {
    if (type != GateType::kMul3) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (!isWireValid(input1) || !isWireValid(input2) || !isWireValid(input3)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<FIn3Gate>(type, input1, input2, input3, output));
    num_wires += 1;

    return output;
  }

  // Function to add a gate with fan-in 4.
  wire_t addGate(GateType type, wire_t input1, wire_t input2, 
                                wire_t input3, wire_t input4) {
    if (type != GateType::kMul4) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (!isWireValid(input1) || !isWireValid(input2) 
        || !isWireValid(input3) || !isWireValid(input4)) {
      throw std::invalid_argument("Invalid wire ID.");
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<FIn4Gate>(type, input1, input2, 
                                          input3, input4, output));
    num_wires += 1;

    return output;
  }

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
    if (type != GateType::kRelu && type != GateType::kMsb
        && type != GateType::kEqz && type != GateType::kLtz) {
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
  wire_t addGate(GateType type, const std::vector<wire_t>& input1,
                 const std::vector<wire_t>& input2) {
    if (type != GateType::kDotprod && type != GateType::kTrdotp) {
      throw std::invalid_argument("Invalid gate type.");
    }

    if (input1.size() != input2.size()) {
      throw std::invalid_argument("Expected same length inputs.");
    }

    for (size_t i = 0; i < input1.size(); ++i) {
      if (!isWireValid(input1[i]) || !isWireValid(input2[i])) {
        throw std::invalid_argument("Invalid wire ID.");
      }
    }

    wire_t output = num_wires;
    gates_.push_back(std::make_shared<SIMDGate>(type, input1, input2, output));
    num_wires += 1;
    return output;
  }

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
  std::vector<std::vector<wire_t>> addMOGate(GateType type, const std::vector<wire_t>& input, const std::vector<std::vector<int>> &permutation,
                                             int nP) {
    if (type != GateType::kAmortzdPnS) {
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

    std::vector<std::vector<wire_t>> output(nP, std::vector<wire_t>(input.size()));
    for (int pid = 0; pid < nP; ++pid) {
      for (int i = 0; i < input.size(); i++) {
        output[pid][i] = pid * nP + i + num_wires;
      }
    }
    gates_.push_back(std::make_shared<SIMDMOGate>(type, 0, input, output, permutation));
    num_wires += nP * input.size();
    return output;
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
        case GateType::kMul3: {
          const auto* g = static_cast<FIn3Gate*>(gate.get());
          size_t gate_depth = std::max(gate_level[g->in1], gate_level[g->in2]);
          gate_depth = std::max(gate_depth, gate_level[g->in3]);
          gate_level[g->out] = gate_depth + 1;
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kMul4: {
          const auto* g = static_cast<FIn4Gate*>(gate.get());
          size_t gate_depth = std::max(gate_level[g->in1], gate_level[g->in2]);
          gate_depth = std::max(gate_depth, gate_level[g->in3]);
          gate_depth = std::max(gate_depth, gate_level[g->in4]);
          gate_level[g->out] = gate_depth + 1;
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

        case GateType::kEqz:
        case GateType::kLtz:
        case GateType::kRelu:
        case GateType::kMsb: {
          const auto* g = static_cast<FIn1Gate*>(gate.get());
          gate_level[g->out] = gate_level[g->in] + 1;
          depth = std::max(depth, gate_level[gate->out]);
          break;
        }

        case GateType::kDotprod:
        case GateType::kTrdotp: {
          const auto* g = static_cast<SIMDGate*>(gate.get());
          size_t gate_depth = 0;
          for (size_t i = 0; i < g->in1.size(); ++i) {
            gate_depth = std::max(
                {gate_level[g->in1[i]], gate_level[g->in2[i]], gate_depth});
          }
          gate_level[g->out] = gate_depth + 1;
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

        case GateType::kAmortzdPnS: {
          const auto* g = static_cast<SIMDMOGate*>(gate.get());
          size_t gate_depth = 0;
          for (size_t i = 0; i < g->in.size(); i++) {
            gate_depth = std::max({gate_level[g->in[i]], gate_depth});
          }
          for (int i = 0; i < g->multi_outs.size(); i++) {
            for (int j = 0; j < g->multi_outs[i].size(); ++j) {
              gate_level[g->multi_outs[i][j]] = gate_depth + 1;
            }
          }
          depth = std::max(depth, gate_level[gate->multi_outs[0][0]]);
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

        default:
          break;
      }
    }

    std::fill(res.count.begin(), res.count.end(), 0);

    std::vector<std::vector<gate_ptr_t>> gates_by_level(depth + 1);
    for (const auto& gate : gates_) {
      res.count[gate->type]++;
      if (gate->type == GateType::kShuffle || gate->type == GateType::kPermAndSh || gate->type == GateType::kPublicPerm) {
        gates_by_level[gate_level[gate->outs[0]]].push_back(gate);
      } else if (gate->type == GateType::kAmortzdPnS) {
        gates_by_level[gate_level[gate->multi_outs[0][0]]].push_back(gate);
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

          case GateType::kMul3: {
            auto* g = static_cast<FIn3Gate*>(gate.get());
            wires[g->out] = wires[g->in1] * wires[g->in2] * wires[g->in3];
            break;
          }

          case GateType::kMul4: {
            auto* g = static_cast<FIn4Gate*>(gate.get());
            wires[g->out] = wires[g->in1] * wires[g->in2] 
                              * wires[g->in3] * wires[g->in4];
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

          case GateType::kLtz: {
            auto* g = static_cast<FIn1Gate*>(gate.get());

            if constexpr (std::is_same_v<R, BoolRing>) {
              wires[g->out] = wires[g->in];
            } else {
              std::vector<BoolRing> bin = bitDecomposeTwo(wires[g->in]);
              wires[g->out] = bin[63].val();
            }
            break;
          }

          case GateType::kRelu: {
            // ReLU gates don't make sense for boolean rings.
            if constexpr (std::is_same_v<R, BoolRing>) {
              throw std::runtime_error("ReLU gates are invalid for BoolRing.");
            } else {
              auto* g = static_cast<FIn1Gate*>(gate.get());
              std::vector<BoolRing> bin = bitDecomposeTwo(wires[g->in]);

              if (bin[63].val())
                wires[g->out] = 0;
              else
                wires[g->out] = wires[g->in];
            }
            break;
          }

          case GateType::kMsb: {
            auto* g = static_cast<FIn1Gate*>(gate.get());

            if constexpr (std::is_same_v<R, BoolRing>) {
              wires[g->out] = wires[g->in];
            } else {
              std::vector<BoolRing> bin = bitDecomposeTwo(wires[g->in]);
              wires[g->out] = bin[63].val();
            }
            break;
          }

          case GateType::kDotprod: {
            auto* g = static_cast<SIMDGate*>(gate.get());
            for (size_t i = 0; i < g->in1.size(); i++) {
              wires[g->out] += wires[g->in1.at(i)] * wires[g->in2.at(i)];
            }
            break;
          }

          case GateType::kTrdotp: {
            // Truncation makes sense only for non-boolean rings.
            if constexpr (std::is_same_v<R, BoolRing>) {
              throw std::runtime_error(
                  "Truncation gates are invalid for BoolRing.");
            } else {
              auto* g = static_cast<SIMDGate*>(gate.get());
              for (size_t i = 0; i < g->in1.size(); i++) {
                auto temp = wires[g->in1.at(i)] * wires[g->in2.at(i)];
                wires[g->out] += temp;
              }
              uint64_t temp = conv<uint64_t>(wires[g->out]);
              temp = temp >> FRACTION;
              wires[g->out] = R(temp);
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

  static Circuit generateParaPrefixOR(int repeat) {
    Circuit circ;
    size_t k = RINGSIZEBITS;
    std::vector<wire_t> input(repeat * k);
    for (int rep = 0; rep < repeat; rep++) {
      for (int i = 0; i < k; i++) {
        input[rep * k + i] = circ.newInputWire();
      }
    }
    std::vector<std::vector<wire_t>> inp_d(repeat, std::vector<wire_t>(k));
    for (int i = 0; i < repeat; i++) {
      for (int j = 0; j < k; j++) {
        inp_d[i][j] = circ.newInputWire();
      }
    }
    R zero = R(0);
    R one = R(1);
    std::vector<wire_t> leveli(repeat * k);
    leveli = std::move(input);
    for (size_t level = 1; level <= log(k) / log(4); level++) {
      std::vector<wire_t> level_next(repeat * k);
      for (size_t j = 1; j <= repeat * k / pow(4, level); j++) {
        size_t p = (j - 1) * pow(4, level);
        size_t q = (j - 1) * pow(4, level) + pow(4, level - 1);
        size_t r = (j - 1) * pow(4, level) + 2 * pow(4, level - 1);
        size_t s = (j - 1) * pow(4, level) + 3 * pow(4, level - 1);
        for (size_t i = 0; i < pow(4, level - 1); i++) {
          level_next[p + i] = circ.addConstOpGate(GateType::kConstAdd, leveli[p + i], zero);
          level_next[q + i] = circ.addGate(GateType::kMul, leveli[q - 1], leveli[q + i]);
          level_next[r + i] = circ.addGate(GateType::kMul3, leveli[q - 1], leveli[r - 1], leveli[r + i]);
          level_next[s + i] = circ.addGate(GateType::kMul4, leveli[q - 1], leveli[r - 1], leveli[s - 1], leveli[s + i]);
        }
      }
      leveli = std::move(level_next);
    }
    // For PrefixOR
    std::vector<std::vector<wire_t>> wv(repeat, std::vector<wire_t>(k));
    for (size_t i = 0; i < repeat; i++) {
      for (size_t j = 0; j < k; j++ ) {
        wv[i][j] = circ.addConstOpGate(GateType::kConstAdd, leveli[i * k + j], one);
      }
    }
    std::vector<std::vector<wire_t>> wz(repeat, std::vector<wire_t>(k));
    for (size_t i = 0; i < repeat; i++) {
      wz[i][0] = circ.addConstOpGate(GateType::kConstAdd, wv[i][0], zero);
      for (size_t j = 1; j < k; j++) {
        wz[i][j] = circ.addGate(GateType::kAdd, wv[i][j], wv[i][j - 1]);
      }
    }
    std::vector<wire_t> inp1(k * repeat), inp2(k * repeat);
    for (size_t i = 0; i < repeat; i++) {
      inp1.insert(inp1.begin(), wz[i].begin(), wz[i].end());
      inp2.insert(inp2.begin(), inp_d[i].begin(), inp_d[i].end());
    }
    wire_t res = circ.addGate(GateType::kDotprod, inp1, inp2);
    circ.setAsOutput(res);
    return circ;
  }

  static Circuit generateMultK() {
    Circuit circ;
    size_t k = RINGSIZEBITS;
    std::vector<wire_t> input(k);
    for (int i = 0; i < k; i++) {
      input[i] = circ.newInputWire();
    }
    std::vector<wire_t> leveli(k);
    leveli = std::move(input);
    for (size_t level = 1; level <= log(k) / log(4); level++) {
      std::vector<wire_t> level_next(k / pow(4, level));
      for (size_t j = 1; j <= k / pow(4, level); j++) {
        level_next[j - 1] = circ.addGate(GateType::kMul4, leveli[(4 * j) - 4], leveli[(4 * j) - 3], leveli[(4 * j) - 2], leveli[(4 * j) - 1]);
      }
      leveli.resize(k / pow(4, level));
      leveli = std::move(level_next);
    }
    circ.setAsOutput(leveli[0]);
    return circ;
  }
};
};  // namespace common::utils
