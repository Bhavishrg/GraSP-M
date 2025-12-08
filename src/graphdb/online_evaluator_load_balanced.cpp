#include "online_evaluator.h"

#include "../utils/helpers.h"

namespace graphdb
{
    // Helper function to reconstruct shares via king party or direct all-to-all
    void OnlineEvaluator::reconstruct(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                         const std::vector<Field>& shares_list, const std::vector<Field>& tags_list,
                         std::vector<Field>& reconstructed_list, std::pair<Field, Field> check,
                         bool via_pking, int latency) {
        int pKing = 1;
        size_t num_shares = shares_list.size();
        reconstructed_list.resize(num_shares);
        
        if (via_pking) {
            // Reconstruction via king party
            if (pid != pKing) {
                network->send(pKing, shares_list.data(), shares_list.size() * sizeof(std::pair<Field, Field>));
                usleep(latency);
                network->recv(pKing, reconstructed_list.data(), reconstructed_list.size() * sizeof(Field));
            } else {
                std::vector<std::vector<Field>> share_recv(nP);
                share_recv[pKing - 1] = shares_list;
                usleep(latency);
                
                // Receive from all parties (not parallelized as recv is blocking)
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        share_recv[p - 1].resize(num_shares);
                        network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(std::pair<Field, Field>));
                    }
                }
                
                // Aggregate shares and reconstruct
                for (size_t i = 0; i < num_shares; ++i) {
                    Field val_sum = Field(0);
                    for (int p = 0; p < nP; ++p) {
                        val_sum += share_recv[p][i];
                    }
                    reconstructed_list[i] = val_sum;
                }
                
                // Send result to all parties (sequential for now to avoid race conditions)
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        network->send(p, reconstructed_list.data(), reconstructed_list.size() * sizeof(Field));
                        network->flush(p);
                    }
                }
            }
        } else {
            // Direct reconstruction (all parties exchange shares)
            std::vector<std::vector<Field>> share_recv(nP);
            share_recv[pid - 1] = shares_list;
            
            // Send to all parties (sequential for now to avoid race conditions)
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    network->send(p, shares_list.data(), shares_list.size() * sizeof(std::pair<Field, Field>));
                }
            }
            
            usleep(latency);
            
            // Receive from all parties
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    share_recv[p - 1].resize(num_shares);
                    network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(std::pair<Field, Field>));
                }
            }
            
            // Aggregate shares
            for (size_t i = 0; i < num_shares; ++i) {
                Field val_sum = Field(0);
                for (int p = 0; p < nP; ++p) {
                    val_sum += share_recv[p][i];
                }
                reconstructed_list[i] = val_sum;
            }
        }
    }

    // Helper function to reconstruct shares towards a designated party
    // Needs to be changed
    // void OnlineEvaluator::reconstructToParty(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
    //                                         const std::vector<Ring>& shares_list,
    //                                         std::vector<Ring>& reconstructed_list,
    //                                         int target_party, int latency) {
    //     size_t num_shares = shares_list.size();
    //     reconstructed_list.resize(num_shares, 0);
        
    //     if (pid == target_party) {
    //         // Target party receives shares from all other parties
    //         std::vector<std::vector<Ring>> share_recv(nP);
    //         share_recv[target_party - 1] = shares_list;
            
    //         usleep(latency);
            
    //         // Receive from all parties except self
    //         for (int p = 1; p <= nP; ++p) {
    //             if (p != target_party) {
    //                 share_recv[p - 1].resize(num_shares);
    //                 network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(Ring));
    //             }
    //         }
            
    //         // Aggregate shares
    //         for (int p = 0; p < nP; ++p) {
    //             for (size_t i = 0; i < num_shares; ++i) {
    //                 reconstructed_list[i] += share_recv[p][i];
    //             }
    //         }
    //     } else {
    //         // Non-target parties send their shares to target party
    //         network->send(target_party, shares_list.data(), shares_list.size() * sizeof(Ring));
    //         network->flush(target_party);
    //     }
    // }

    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit preproc,
                                     common::utils::LevelOrderedCircuit circ,
                                     int threads, int seed, int latency, bool use_pking)
        : nP_(nP),
          id_(id),
          latency_(latency),
          use_pking_(use_pking),
          rgen_(id, seed),
          network_(std::move(network)),
          preproc_(std::move(preproc)),
          circ_(std::move(circ)),
          wires_(circ.num_wires)
    {
        tpool_ = std::make_shared<common::utils::ThreadPool>(threads);
    }

    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit preproc,
                                     common::utils::LevelOrderedCircuit circ,
                                     std::shared_ptr<common::utils::ThreadPool> tpool, int seed, int latency, bool use_pking)
        : nP_(nP),
          id_(id),
          latency_(latency),
          use_pking_(use_pking),
          rgen_(id, seed),
          network_(std::move(network)),
          preproc_(std::move(preproc)),
          circ_(std::move(circ)),
          tpool_(std::move(tpool)),
          wires_(circ.num_wires) {}

    
    void OnlineEvaluator::setInputs(const std::unordered_map<common::utils::wire_t, Field> &inputs) {
        // Input gates have depth 0
        // for (auto &g : circ_.gates_by_level[0]) {
        //     if (g->type == common::utils::GateType::kInp) {
        //         auto *pre_input = static_cast<PreprocInput<Ring> *>(preproc_.gates[g->out].get());
        //         auto pid = pre_input->pid;
        //         if (id_ != 0) {
        //             if (pid == id_) {
        //                 Ring accumulated_val = Ring(0);
        //                 for (size_t i = 1; i <= nP_; i++) {
        //                     if (i != pid) {
        //                         Ring rand_sh;
        //                         network_->send(pid, output_shares[id_ - 1].data(), output_shares[id_ - 1].size() * sizeof(Ring));
        //                     }
        //                 }
        //                 wires_[g->out] = inputs.at(g->out) - accumulated_val;
        //             } else {
        //                 rgen_.pi(id_).random_data(&wires_[g->out], sizeof(Ring));
        //             }
        //         }
        //     }
        // }

        if (id_ == 0) { return; }

        std::vector<size_t> gate_size_by_owner(nP_ + 1);  // +1 because parties are numbered 1 to nP
        std::vector<Field> masked_inputs;
        std::vector<common::utils::wire_t> input_wires;
        gate_size_by_owner[0] = 0;

        for (auto &g : circ_.gates_by_level[0]) {
            if (g->type == common::utils::GateType::kInp) {
                auto *pre_input = static_cast<PreprocInput *>(preproc_.gates[g->out].get());
                gate_size_by_owner[pre_input->pid]++;
                if (pre_input->pid == id_) {
                    // This party owns this input
                    Field masked_val = inputs.at(g->out) + pre_input->r;
                    masked_inputs.push_back(masked_val);
                }
            }
        }

        // Batch send masked inputs to all other parties
        std::vector<std::vector<Field>> received_masked_inputs(nP_);
        received_masked_inputs[0] = std::vector<Field>();  // No inputs from party 0
        for (int p = 1; p <= nP_; ++p) {
            if (gate_size_by_owner[p] > 0) {
                if (p == id_) {
                    for (int q = 1; q <= nP_; ++q) {
                        if (q != id_) {
                            network_->send(q, masked_inputs.data(), gate_size_by_owner[p] * sizeof(Field));
                        }
                    }
                } else {
                    received_masked_inputs[p].resize(gate_size_by_owner[p]);
                    network_->recv(p, received_masked_inputs[p].data(), gate_size_by_owner[p] * sizeof(Field));
                }                
            }
        }
        usleep(latency_);

        // Finally, set the wire values
        std::vector<size_t> recv_idx(nP_, 0);  // Track index for each party's received values
        
        for (auto &g : circ_.gates_by_level[0]) {
            if (g->type == common::utils::GateType::kInp) {
                auto *pre_input = static_cast<PreprocInput *>(preproc_.gates[g->out].get());
                AuthAddShare share_inp;
                Field share_alpha = share_inp.keySh();

                if (pre_input->pid == id_) {
                    // This party owns this input - use its own masked value
                    Field masked_val = inputs.at(g->out) + pre_input->r;
                    share_inp.pushValue(masked_val - pre_input->share_r.valueAt());
                    share_inp.pushTag(masked_val * share_alpha - pre_input->share_r.tagAt());
                    wires_[g->out] = share_inp;
                } else {
                    // This party does not own this input - use received masked value
                    int owner = pre_input->pid;
                    Field masked_val = received_masked_inputs[owner][recv_idx[owner]];
                    recv_idx[owner]++;
                    share_inp.pushValue(-pre_input->share_r.valueAt());
                    share_inp.pushTag(masked_val * share_alpha - pre_input->share_r.tagAt());
                    wires_[g->out] = share_inp;
                }
            }
        }

    }

    void OnlineEvaluator::setRandomInputs() {
        // Input gates have depth 0.
        for (auto &g : circ_.gates_by_level[0]) {
            // if (g->type == common::utils::GateType::kInp) {
            //     rgen_.pi(id_).random_data(&wires_[g->out], sizeof(Ring));
            // }

            if (g->type == common::utils::GateType::kInp) {
                auto *pre_input = static_cast<PreprocInput *>(preproc_.gates[g->out].get());
                wires_[g->out] = pre_input->share_r;
            }
        }
    }

    void OnlineEvaluator::recEvaluate(const std::vector<common::utils::FIn1Gate> &rec_gates) {
        if (id_ == 0) { return; }
        size_t num_rec_gates = rec_gates.size();
        std::vector<std::pair<Field, Field>> shares_to_send(num_rec_gates);

        // Gather shares to reconstruct
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            auto &rec_gate = rec_gates[i];
            shares_to_send[i] = std::make_pair(wires_[rec_gate.in].valueAt(), wires_[rec_gate.in].tagAt());
        }

        // Reconstruct the values
        std::vector<Field> reconstructed(num_rec_gates);
        std::pair<Field, Field> check_val = std::make_pair(check.valueAt(), check.tagAt());
        
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, check_val, use_pking_, latency_);

        // Store reconstructed values in output wires
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            AuthAddShare reconstructed_share;
            reconstructed_share.pushValue(reconstructed[i]);
            reconstructed_share.pushTag(Field(0));
            wires_[rec_gates[i].out] = reconstructed_share;
        }
    }

    void OnlineEvaluator::multEvaluate(const std::vector<common::utils::FIn2Gate> &mult_gates) {
        if (id_ == 0) { return; }
        
        size_t num_mult_gates = mult_gates.size();
        if (num_mult_gates == 0) { return; }
        
        // Step 1: Compute shares of u and v and corresponding tags for each multiplication gate
        // For multiplication z = x * y using Beaver triples (a, b, c) where c = a * b:
        // - u = x - a (each party computes their share)
        // - v = y - b (each party computes their share)
        // - Reconstruct u and v to get actual values
        // - Compute z = u*v + u*b + v*a + c
        
        //Vector of shares of u and v
        std::vector<Field> u_shares(num_mult_gates);
        std::vector<Field> v_shares(num_mult_gates);

        //Vector of tags of u and v
        std::vector<Field> u_tags(num_mult_gates);
        std::vector<Field> v_tags(num_mult_gates);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_mult_gates; ++i) {
            auto &mult_gate = mult_gates[i];
            auto *pre_out = static_cast<PreprocMultGate*>(preproc_.gates[mult_gate.out].get());
            
            // Compute this party's share of u and v
            u_shares[i] = wires_[mult_gate.in1].valueAt() - pre_out->triple_a.valueAt();
            v_shares[i] = wires_[mult_gate.in2].valueAt() - pre_out->triple_b.valueAt();

            // Compute this party's share of tags of u and v
            u_tags[i] = wires_[mult_gate.in1].tagAt() - pre_out->triple_a.tagAt();
            v_tags[i] = wires_[mult_gate.in2].tagAt() - pre_out->triple_b.tagAt();
        }
        
        // Step 2: Reconstruct u and v values
        std::vector<Field> u_reconstructed(num_mult_gates);
        std::vector<Field> v_reconstructed(num_mult_gates);
        
        // Prepare shares and tags to send (interleaved: u0, v0, u1, v1, ...)
        std::vector<Field> shares_to_send(2 * num_mult_gates);
        std::vector<Field> tags_to_send(2 * num_mult_gates);
        for (size_t i = 0; i < num_mult_gates; ++i) {
            shares_to_send[2*i] = u_shares[i];
            tags_to_send[2*i] = u_tags[i];
            shares_to_send[2*i + 1] = v_shares[i];
            tags_to_send[2*i + 1] = v_tags[i];
        }
        
        std::vector<Field> reconstructed(2 * num_mult_gates);
        reconstruct(nP_, id_, network_, shares_to_send, tags_to_send, reconstructed, use_pking_, latency_);
        
        // Unpack reconstructed values
        for (size_t i = 0; i < num_mult_gates; ++i) {
            u_reconstructed[i] = reconstructed[2*i];
            v_reconstructed[i] = reconstructed[2*i + 1];
        }
        
        // Step 3: Compute multiplication result using Beaver triple formula
        // z = u*v + u*b + v*a + c (where only c is a share, rest are public values)
        #pragma omp parallel for
        for (size_t i = 0; i < num_mult_gates; ++i) {
            auto &mult_gate = mult_gates[i];
            auto *pre_out = static_cast<PreprocMultGate*>(preproc_.gates[mult_gate.out].get());
            
            Field u = u_reconstructed[i];
            Field v = v_reconstructed[i];
            Field a = pre_out->triple_a.valueAt();
            Field b = pre_out->triple_b.valueAt();
            Field c = pre_out->triple_c.valueAt();
            
            // Beaver triple formula: z = u*v + u*b + v*a + c
            if (id_ == 1){
                wires_[mult_gate.out].pushValue(u * v + u * b + v * a + c);
            }
            else{
                wires_[mult_gate.out].pushValue(u * b + v * a + c);
            }
        }
    }

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }

        // std::vector<common::utils::FIn2Gate> mult_gates;
        // std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::FIn1Gate> rec_gates;
        // std::vector<common::utils::SIMDOGate> shuffle_gates;
        // std::vector<common::utils::SIMDOGate> permAndSh_gates;
        // std::vector<common::utils::SIMDOGate> amortzdPnS_gates;
        // std::vector<common::utils::SIMDOGate> rewire_gates;

        // First pass: collect the multi-round gates so their batched handlers can run once.
        for (auto &gate : circ_.gates_by_level[depth]) {
            switch (gate->type) {
                // case common::utils::GateType::kMul: {
                //     auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                //     mult_gates.push_back(*g);
                //     break;
                // }
                // case common::utils::GateType::kEqz: {
                //     auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                //     eqz_gates.push_back(*g);
                //     break;
                // }
                case common::utils::GateType::kRec: {
                    auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                    rec_gates.push_back(*g);
                    break;
                }
                // case common::utils::GateType::kShuffle: {
                //     auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                //     shuffle_gates.push_back(*g);
                //     break;
                // }
                // case common::utils::GateType::kPermAndSh: {
                //     auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                //     permAndSh_gates.push_back(*g);
                //     break;
                // }

                // case common::utils::GateType::kAmortzdPnS: {
                //     auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                //     amortzdPnS_gates.push_back(*g);
                //     break;
                // }

                // case common::utils::GateType::kRewire: {
                //     auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                //     // Rewire gates are processed immediately
                //     rewire_gates.push_back(*g);
                //     break;
                // }

                default:
                    break;
            }
        }

        // if (!mult_gates.empty()) {  }
        // if (!eqz_gates.empty()) {  }
        if (!rec_gates.empty()) { recEvaluate(rec_gates); }
        // if (!shuffle_gates.empty()) {  }
        // if (!permAndSh_gates.empty()) {  }
        // if (!amortzdPnS_gates.empty()) {  }
        // if (!rewire_gates.empty()) {  }
        
        // Second pass: handle locally evaluable gates.
        for (auto &gate : circ_.gates_by_level[depth]) {
            switch (gate->type) {
                case common::utils::GateType::kAdd: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    wires_[g->out] = wires_[g->in1] + wires_[g->in2];
                    break;
                }
                case common::utils::GateType::kSub: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    wires_[g->out] = wires_[g->in1] - wires_[g->in2];
                    break;
                }
                case common::utils::GateType::kConstAdd: {
                    auto *g = static_cast<common::utils::ConstOpGate<Field> *>(gate.get());
                    Field alpha = wires_[g->in].keySh();
                    if (id_ == 1) {
                        wires_[g->out].pushValue(wires_[g->in].valueAt() + g->cval);
                        wires_[g->out].pushTag(wires_[g->in].tagAt() + alpha*g->cval);
                    } else {
                        wires_[g->out] = wires_[g->in];
                        wires_[g->out].pushTag(wires_[g->in].tagAt() + alpha*g->cval);
                    }
                    break;
                }
                case common::utils::GateType::kConstMul: {
                    auto *g = static_cast<common::utils::ConstOpGate<Field> *>(gate.get());
                    wires_[g->out].pushValue(wires_[g->in].valueAt()*g->cval);
                    wires_[g->out].pushTag(wires_[g->in].tagAt()*g->cval);
                    break;
                }
                default:
                    break;
            }
        }
    }

    std::vector<Field> OnlineEvaluator::getOutputs() {
        std::vector<Field> outvals(circ_.outputs.size());
        if (circ_.outputs.empty()) {
            return outvals;
        }
        if (id_ != 0) {
            std::vector<std::vector<Field>> output_shares(nP_, std::vector<Field>(circ_.outputs.size()));
            for (size_t i = 0; i < circ_.outputs.size(); ++i) {
                auto wout = circ_.outputs[i];
                output_shares[id_ - 1][i] = wires_[wout].valueAt();
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->send(pid, output_shares[id_ - 1].data(), output_shares[id_ - 1].size() * sizeof(Field));
                }
            }
            usleep(latency_);
            // #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->recv(pid, output_shares[pid - 1].data(), output_shares[pid - 1].size() * sizeof(Field));
                }
            }
            for (size_t i = 0; i < circ_.outputs.size(); ++i) {
                Field outmask = Field(0);
                for (int pid = 1; pid <= nP_; ++pid) {
                    outmask += output_shares[pid - 1][i];
                }
                outvals[i] = outmask;
            }
        }
        return outvals;
    }

    std::vector<Field> OnlineEvaluator::evaluateCircuit(const std::unordered_map<common::utils::wire_t, Field> &inputs) {
        setInputs(inputs);
        for (size_t i = 0; i < circ_.gates_by_level.size(); ++i) {
            evaluateGatesAtDepth(i);
        }
        return getOutputs();
    }

}; // namespace graphdb