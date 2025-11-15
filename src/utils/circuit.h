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
  kDeleteWires,
  kGroupwiseIndex,
  kGroupwisePropagate,
  kSort,
  kRewire,
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
  size_t vec_size{0};  // Metadata: size of vector for SIMD operations

  SIMDOGate() = default;
  SIMDOGate(GateType type, int owner, std::vector<wire_t> in, std::vector<wire_t> out, std::vector<std::vector<int>> permutation, size_t vec_size = 0);
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

  // Create a constant value wire
  // Party 1 will hold the constant value, all other parties will hold 0
  wire_t newConstWire(R cval, int pid) {
    wire_t val;
    if (pid == 1) {
      val = cval;
    } else {
      val = static_cast<R>(0);
    } 
    return val;
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



  // Add a Group-wise Index gate
  // Takes key vector (binary) and value vector, returns (ind, key, v)
  // where ind[i] = number of elements in the group before position i
  std::tuple<std::vector<wire_t>, std::vector<wire_t>, std::vector<wire_t>> addGroupwiseIndexGate(
      const std::vector<wire_t>& key_vector,
      const std::vector<wire_t>& v_vector,
      const std::vector<std::vector<int>>& permutation) {
    
    size_t vec_size = key_vector.size();
    
    if (v_vector.size() != vec_size) {
      throw std::invalid_argument("Value vector must have the same size as key vector.");
    }
    
    // Validate input wires
    for (size_t i = 0; i < vec_size; i++) {
      if (!isWireValid(key_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in key_vector.");
      }
      if (!isWireValid(v_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in v_vector.");
      }
    }
    
    // Create output wires: vec_size for ind, vec_size for key, vec_size for v
    std::vector<wire_t> output_ind(vec_size);
    std::vector<wire_t> output_key(vec_size);
    std::vector<wire_t> output_v(vec_size);
    
    for (size_t i = 0; i < vec_size; i++) {
      output_ind[i] = num_wires + i;
      output_key[i] = num_wires + vec_size + i;
      output_v[i] = num_wires + 2 * vec_size + i;
    }
    
    // Create input vector [key0, key1, ..., keyn, v0, v1, ..., vn]
    std::vector<wire_t> input(2 * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      input[i] = key_vector[i];
      input[vec_size + i] = v_vector[i];
    }
    
    // Create output vector for the gate
    std::vector<wire_t> output(3 * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      output[i] = output_ind[i];
      output[vec_size + i] = output_key[i];
      output[2 * vec_size + i] = output_v[i];
    }
    
    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kGroupwiseIndex, 0, input, output, permutation));
    num_wires += 3 * vec_size;
    
    return {output_ind, output_key, output_v};
  }

  // Add a Group-wise Propagate gate
  // Takes T1 (key1, v1) and T2 (key2), returns (key2, v_out)
  // where v_out propagates values from T1 based on group structure in T2
  std::pair<std::vector<wire_t>, std::vector<wire_t>> addGroupwisePropagateGate(
      const std::vector<wire_t>& key1_vector,
      const std::vector<wire_t>& v1_vector,
      const std::vector<wire_t>& key2_vector,
      const std::vector<std::vector<int>>& permutation) {
    
    size_t t1_vec_size = key1_vector.size();
    size_t t2_vec_size = key2_vector.size();
    
    if (v1_vector.size() != t1_vec_size) {
      throw std::invalid_argument("Value vector must have the same size as key1 vector.");
    }
    
    // Validate input wires
    for (size_t i = 0; i < t1_vec_size; i++) {
      if (!isWireValid(key1_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in key1_vector.");
      }
      if (!isWireValid(v1_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in v1_vector.");
      }
    }
    for (size_t i = 0; i < t2_vec_size; i++) {
      if (!isWireValid(key2_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in key2_vector.");
      }
    }
    
    // Create output wires: t2_vec_size for key2, t2_vec_size for v_out
    std::vector<wire_t> output_key2(t2_vec_size);
    std::vector<wire_t> output_v(t2_vec_size);
    
    for (size_t i = 0; i < t2_vec_size; i++) {
      output_key2[i] = num_wires + i;
      output_v[i] = num_wires + t2_vec_size + i;
    }
    
    // Create input vector [key1_0,...,key1_n1, v1_0,...,v1_n1, key2_0,...,key2_n2]
    std::vector<wire_t> input(2 * t1_vec_size + t2_vec_size);
    for (size_t i = 0; i < t1_vec_size; i++) {
      input[i] = key1_vector[i];
      input[t1_vec_size + i] = v1_vector[i];
    }
    for (size_t i = 0; i < t2_vec_size; i++) {
      input[2 * t1_vec_size + i] = key2_vector[i];
    }
    
    // Create output vector [key2_0,...,key2_n2, v_out_0,...,v_out_n2]
    std::vector<wire_t> output(2 * t2_vec_size);
    for (size_t i = 0; i < t2_vec_size; i++) {
      output[i] = output_key2[i];
      output[t2_vec_size + i] = output_v[i];
    }
    
    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kGroupwisePropagate, 0, input, output, permutation));
    num_wires += 2 * t2_vec_size;
    
    return {output_key2, output_v};
  }

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

  // Add a Sort gate that takes bit-decomposed 32-bit integers and returns sorted output
  // Input format: [bit0_elem0, bit1_elem0, ..., bit31_elem0, bit0_elem1, ..., bit31_elem(n-1)]
  // Each consecutive 32 wires represent one 32-bit value in bit-decomposed form
  // Returns sorted values in the same bit-decomposed format
  std::vector<wire_t> addSortGate(
      const std::vector<wire_t>& bit_decomposed_input,
      const std::vector<std::vector<int>>& permutation) {
    
    // Input should be divisible by 32 (32 bits per element)
    if (bit_decomposed_input.size() % 32 != 0) {
      throw std::invalid_argument("Input size must be divisible by 32 (32 bits per element).");
    }
    
    size_t total_wires = bit_decomposed_input.size();
    size_t vec_size = total_wires / 32;  // Number of 32-bit integers
    
    if (vec_size == 0) {
      throw std::invalid_argument("At least one 32-bit element is required.");
    }
    
    // Validate input wires
    for (size_t i = 0; i < total_wires; i++) {
      if (!isWireValid(bit_decomposed_input[i])) {
        throw std::invalid_argument("Invalid wire ID in bit_decomposed_input.");
      }
    }
    
    // Create output wires: 32 * vec_size wires (same structure as input)
    std::vector<wire_t> sorted_output(total_wires);
    for (size_t i = 0; i < total_wires; i++) {
      sorted_output[i] = num_wires + i;
    }
    
    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kSort, 0, bit_decomposed_input, sorted_output, permutation));
    num_wires += total_wires;
    
    return sorted_output;
  }

  // Add a Rewire gate that applies a public permutation based on a position map
  // Takes a position map vector and any number of payload vectors as input
  // Outputs permuted payload wires based on the position map
  // 
  // Gate behavior (similar to lines 678-690 in compactEvaluate):
  //   For each position i: if position_map[i] = idx_perm, then output[idx_perm] = payload[i]
  //
  // Input format: [pos_map_0, ..., pos_map_n, p1_0, ..., p1_n, p2_0, ..., p2_n, ...]
  // Output format: [p1_out_0, ..., p1_out_n, p2_out_0, ..., p2_out_n, ...]
  std::vector<std::vector<wire_t>> addRewireGate(
      const std::vector<wire_t>& position_map,
      const std::vector<std::vector<wire_t>>& payload_vectors) {
    
    size_t vec_size = position_map.size();
    size_t num_payloads = payload_vectors.size();
    
    if (num_payloads == 0) {
      throw std::invalid_argument("At least one payload vector is required.");
    }
    
    // Validate all payload vectors have same size as position_map
    for (size_t p = 0; p < num_payloads; ++p) {
      if (payload_vectors[p].size() != vec_size) {
        throw std::invalid_argument("All payload vectors must have the same size as position_map.");
      }
    }
    
    // Validate input wires
    for (size_t i = 0; i < vec_size; i++) {
      if (!isWireValid(position_map[i])) {
        throw std::invalid_argument("Invalid wire ID in position_map.");
      }
      for (size_t p = 0; p < num_payloads; ++p) {
        if (!isWireValid(payload_vectors[p][i])) {
          throw std::invalid_argument("Invalid wire ID in payload vector.");
        }
      }
    }
    
    // Create output wires: vec_size for each payload
    std::vector<std::vector<wire_t>> payload_outputs(num_payloads, std::vector<wire_t>(vec_size));
    
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        payload_outputs[p][i] = num_wires + vec_size * p + i;
      }
    }
    
    // Create input vector [pos_map_0, ..., pos_map_n, p1_0, ..., p1_n, p2_0, ...]
    std::vector<wire_t> input((1 + num_payloads) * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      input[i] = position_map[i];
    }
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        input[vec_size * (p + 1) + i] = payload_vectors[p][i];
      }
    }
    
    // Create output vector for the gate
    std::vector<wire_t> output(num_payloads * vec_size);
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        output[vec_size * p + i] = payload_outputs[p][i];
      }
    }
    
    // Empty permutation vector since the permutation is determined at runtime from position_map
    std::vector<std::vector<int>> empty_permutation;
    // Pass vec_size as metadata to the gate constructor
    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kRewire, 0, input, output, empty_permutation, vec_size));
    num_wires += num_payloads * vec_size;
    
    return payload_outputs;
  }


  // Add a Delete Wires gate that takes a delete mask `del` and multiple payload
  // vectors. Indices where `del[i] == 1` will be removed from the payloads.
  // The gate reserves output wires equal to the input vector size. At runtime
  // the front of each output vector will contain the compacted payloads and
  // unused tail positions may be ignored by subsequent logic. The original
  // vector size is passed through the gate metadata (`vec_size`).
  // The permutation parameter determines the size of del and payload vectors.
  // Returns: (keep_indices_wire, payload_outputs)
  // keep_indices_wire holds the actual keep_indices array (only on party 1, others have 0)
  std::pair<wire_t, std::vector<std::vector<wire_t>>> addDeleteWiresGate(
      const std::vector<wire_t>& del_vector,
      const std::vector<std::vector<wire_t>>& payload_vectors,
      const std::vector<std::vector<int>>& permutation) {
    
    if (permutation.size() == 0) {
      throw std::invalid_argument("No permutation passed.");
    }

    size_t vec_size = permutation[0].size();
    size_t num_payloads = payload_vectors.size();

    if (num_payloads == 0) {
      throw std::invalid_argument("At least one payload vector is required.");
    }

    if (del_vector.size() != vec_size) {
      throw std::invalid_argument("del_vector size must match permutation size.");
    }

    for (size_t i = 0; i < permutation.size(); ++i) {
      if (permutation[i].size() != vec_size) {
        throw std::invalid_argument("Permutation size mismatch.");
      }
    }

    for (size_t p = 0; p < num_payloads; ++p) {
      if (payload_vectors[p].size() != vec_size) {
        throw std::invalid_argument("All payload vectors must have the same size as permutation.");
      }
    }

    for (size_t i = 0; i < vec_size; i++) {
      if (!isWireValid(del_vector[i])) {
        throw std::invalid_argument("Invalid wire ID in del_vector.");
      }
      for (size_t p = 0; p < num_payloads; ++p) {
        if (!isWireValid(payload_vectors[p][i])) {
          throw std::invalid_argument("Invalid wire ID in payload vector.");
        }
      }
    }

    // Allocate output wires: 1 wire for keep_indices, vec_size for each payload
    wire_t keep_indices_wire = num_wires;
    
    std::vector<std::vector<wire_t>> payload_outputs(num_payloads, std::vector<wire_t>(vec_size));
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        payload_outputs[p][i] = num_wires + 1 + vec_size * p + i;
      }
    }

    // Create input vector [del_0,...,del_n, p1_0,...,p1_n, p2_0,...]
    std::vector<wire_t> input((1 + num_payloads) * vec_size);
    for (size_t i = 0; i < vec_size; i++) {
      input[i] = del_vector[i];
    }
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        input[vec_size * (p + 1) + i] = payload_vectors[p][i];
      }
    }

    // Flatten outputs into a single vector for the gate constructor
    // Output format: [keep_indices_wire, p1_0,...,p1_n, p2_0,...,p2_n, ...]
    std::vector<wire_t> output(1 + num_payloads * vec_size);
    output[0] = keep_indices_wire;
    for (size_t p = 0; p < num_payloads; ++p) {
      for (size_t i = 0; i < vec_size; i++) {
        output[1 + vec_size * p + i] = payload_outputs[p][i];
      }
    }

    gates_.push_back(std::make_shared<SIMDOGate>(GateType::kDeleteWires, 0, input, output, permutation, vec_size));
    num_wires += 1 + num_payloads * vec_size;

    return {keep_indices_wire, payload_outputs};
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

        case GateType::kDeleteWires: {
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

        case GateType::kGroupwiseIndex: {
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

        case GateType::kGroupwisePropagate: {
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

        case GateType::kSort: {
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

        case GateType::kRewire: {
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
      if (gate->type == GateType::kShuffle || gate->type == GateType::kPermAndSh || gate->type == GateType::kPublicPerm || gate->type == GateType::kCompact || gate->type == GateType::kDeleteWires || gate->type == GateType::kGroupwiseIndex || gate->type == GateType::kGroupwisePropagate || gate->type == GateType::kSort || gate->type == GateType::kRewire) {
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
