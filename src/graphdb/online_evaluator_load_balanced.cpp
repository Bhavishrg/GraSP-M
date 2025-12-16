#include "online_evaluator.h"

#include "../utils/helpers.h"

namespace graphdb
{
    // Helper function to reconstruct shares via king party or direct all-to-all
    void OnlineEvaluator::reconstruct(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                                     std::vector<AuthAddShare>& shares_list,
                                     std::vector<Field>& reconstructed_list, AuthAddShare& check,
                                     bool via_pking, int latency,
                                     std::vector<AuthAddShare>* tag_shares_list,
                                     std::vector<Field>* tag_reconstructed_list) {
        if (pid == 0) {return;}

        int pKing = 1;
        size_t num_shares = shares_list.size();
        reconstructed_list.resize(num_shares);
        
        size_t tag_num_shares = 0;
        bool has_second_list = (tag_shares_list != nullptr && tag_reconstructed_list != nullptr);
        if (has_second_list) {
            tag_num_shares = tag_shares_list->size();
            tag_reconstructed_list->resize(num_shares);
        }
        
        size_t total_shares = num_shares + tag_num_shares;

        std::vector<Field> shares_to_send(total_shares);
        for (size_t i = 0; i < num_shares; ++i) {
            shares_to_send[i] = shares_list[i].valueAt();
        }
        if (has_second_list) {
            // tags are reconstructed in the second list
            for (size_t i = 0; i < tag_num_shares; ++i) {
                shares_to_send[num_shares + i] = (*tag_shares_list)[i].tagAt();
            }
        }
        
        if (via_pking) {
            // Reconstruction via king party
            if (pid != pKing) {
                network->send(pKing, shares_to_send.data(), shares_to_send.size() * sizeof(Field));
                usleep(latency);
                
                std::vector<Field> all_reconstructed(total_shares);
                network->recv(pKing, all_reconstructed.data(), all_reconstructed.size() * sizeof(Field));
                
                // Split into two lists
                for (size_t i = 0; i < num_shares; ++i) {
                    reconstructed_list[i] = all_reconstructed[i];
                }
                if (has_second_list) {
                    for (size_t i = 0; i < tag_num_shares; ++i) {
                        (*tag_reconstructed_list)[i] = all_reconstructed[num_shares + i];
                    }
                }
            } else {
                std::vector<std::vector<Field>> share_recv(nP);
                share_recv[pKing - 1] = shares_to_send;
                
                usleep(latency);
                // Receive from all parties (not parallelized as recv is blocking)
                // #pragma omp parallel for
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        share_recv[p - 1].resize(total_shares);
                        network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(Field));
                    }
                }
                
                // Aggregate shares for first list
                for (int p = 0; p < nP; ++p) {
                    for (size_t i = 0; i < num_shares; ++i) {
                        reconstructed_list[i] += share_recv[p][i];
                    }
                }
                
                // Aggregate shares for second list if present
                if (has_second_list) {
                    for (int p = 0; p < nP; ++p) {
                        for (size_t i = 0; i < tag_num_shares; ++i) {
                            (*tag_reconstructed_list)[i] += share_recv[p][num_shares + i];
                        }
                    }
                }
                
                // Send result to all parties (sequential for now to avoid race conditions)
                std::vector<Field> all_reconstructed(total_shares);
                for (size_t i = 0; i < num_shares; ++i) {
                    all_reconstructed[i] = reconstructed_list[i];
                }
                if (has_second_list) {
                    for (size_t i = 0; i < tag_num_shares; ++i) {
                        all_reconstructed[num_shares + i] = (*tag_reconstructed_list)[i];
                    }
                }
                
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        network->send(p, all_reconstructed.data(), all_reconstructed.size() * sizeof(Field));
                        network->flush(p);
                    }
                }
            }
        } else {
            // Direct reconstruction (all parties exchange shares)
            std::vector<std::vector<Field>> share_recv(nP);
            share_recv[pid - 1] = shares_to_send;
            
            // Send to all parties (sequential for now to avoid race conditions)
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    network->send(p, shares_to_send.data(), shares_to_send.size() * sizeof(Field));
                }
            }
            
            usleep(latency);
            
            // Receive from all parties
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    share_recv[p - 1].resize(total_shares);
                    network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(Field));
                }
            }
            
            // Aggregate shares for first list
            for (int p = 0; p < nP; ++p) {
                for (size_t i = 0; i < num_shares; ++i) {
                    reconstructed_list[i] += share_recv[p][i];
                }
            }
            
            // Aggregate shares for second list if present
            if (has_second_list) {
                for (int p = 0; p < nP; ++p) {
                    for (size_t i = 0; i < tag_num_shares; ++i) {
                        (*tag_reconstructed_list)[i] += share_recv[p][num_shares + i];
                    }
                }
            }
        }

        // Check computation would go here if needed
        Field alpha_sh = check.keySh();
        Field alpha_inv_sh = check.invkeySh();


        for (size_t i = 0; i < num_shares; ++i) {
            Field rand;
            randomizeZZp(rgen_.all_minus_0(), rand, sizeof(Field));

            Field new_value;

            if (pid == 1) {
                new_value = check.valueAt() + rand*(reconstructed_list[i] - shares_list[i].valueAt());
            } else {
                new_value =  check.valueAt() + rand*(Field(0) - shares_list[i].valueAt());
            }
            // Field new_value = check.valueAt() + reconstructed_list[i] - shares_list[i].valueAt();
            Field new_tag = check.tagAt() + rand*(reconstructed_list[i] * alpha_sh - shares_list[i].tagAt());
            check.pushValue(new_value);
            check.pushTag(new_tag);

        }
        
        // Process second list if present
        if (has_second_list) {
            for (size_t i = 0; i < tag_num_shares; ++i) {
                Field rand;
                randomizeZZp(rgen_.all_minus_0(), rand, sizeof(Field));

                Field new_value;
                Field new_tag;
                new_value = check.valueAt() + rand*((*tag_reconstructed_list)[i] * alpha_inv_sh - (*tag_shares_list)[i].valueAt());
                if (pid == 1) {
                    new_tag = check.tagAt() + rand*((*tag_reconstructed_list)[i] - (*tag_shares_list)[i].tagAt());
                } else {
                    new_tag =  check.tagAt() + rand*(Field(0) - (*tag_shares_list)[i].tagAt());
                }
                
                check.pushValue(new_value);
                check.pushTag(new_tag);
            }
        }
    }

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
          wires_(circ.num_wires),
          check(Field(0), Field(0))
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
          wires_(circ.num_wires),
          check(Field(0), Field(0)) {}

    
    void OnlineEvaluator::setInputs(const std::unordered_map<common::utils::wire_t, Field> &inputs) {

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

            if (g->type == common::utils::GateType::kInp) {
                auto *pre_input = static_cast<PreprocInput *>(preproc_.gates[g->out].get());
                wires_[g->out] = pre_input->share_r;
            }
        }
    }

    void OnlineEvaluator::multEvaluate(const std::vector<common::utils::FIn2Gate> &mult_gates) {
        if (id_ == 0) { return; }
        
        size_t num_mult_gates = mult_gates.size();
        if (num_mult_gates == 0) { return; }
        
        // Step 1: Compute shares of u and v for each multiplication gate
        // For multiplication z = x * y using Beaver triples (a, b, c) where c = a * b:
        // - u = x - a (each party computes their share)
        // - v = y - b (each party computes their share)
        // - Reconstruct u and v to get actual values
        // - Compute z = u*v + u*b + v*a + c
        
        std::vector<AuthAddShare> u_shares(num_mult_gates);
        std::vector<AuthAddShare> v_shares(num_mult_gates);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_mult_gates; ++i) {
            auto &mult_gate = mult_gates[i];
            auto *pre_out = static_cast<PreprocMultGate *>(preproc_.gates[mult_gate.out].get());
            
            // Compute this party's share of u and v
            u_shares[i] = wires_[mult_gate.in1] - pre_out->triple_a;
            v_shares[i] = wires_[mult_gate.in2] - pre_out->triple_b;
        }
        
        std::cout << "doen here " << std::endl;

        // Step 2: Reconstruct u and v values
        std::vector<Field> u_reconstructed(num_mult_gates);
        std::vector<Field> v_reconstructed(num_mult_gates);
        
        // Prepare shares to send (interleaved: u0, v0, u1, v1, ...)
        std::vector<AuthAddShare> shares_to_send(2 * num_mult_gates);
        for (size_t i = 0; i < num_mult_gates; ++i) {
            shares_to_send[2*i] = u_shares[i];
            shares_to_send[2*i + 1] = v_shares[i];
        }
        
        std::vector<Field> reconstructed(2 * num_mult_gates);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, check, use_pking_, latency_);
        
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
            auto *pre_out = static_cast<PreprocMultGate *>(preproc_.gates[mult_gate.out].get());
            
            Field u = u_reconstructed[i];
            Field v = v_reconstructed[i];
            Field a = pre_out->triple_a.valueAt();
            Field b = pre_out->triple_b.valueAt();
            Field c = pre_out->triple_c.valueAt();

            Field tag_a = pre_out->triple_a.tagAt();
            Field tag_b = pre_out->triple_b.tagAt();
            Field tag_c = pre_out->triple_c.tagAt();

            Field key_sh = pre_out->triple_a.keySh();
            
            // Beaver triple formula: z = u*v + u*b + v*a + c
            if (id_ == 1){
                wires_[mult_gate.out].pushValue(u * v + u * b + v * a + c);
                wires_[mult_gate.out].pushTag(u * v*key_sh + u * tag_b + v * tag_a + tag_c);
            }
            else{
                wires_[mult_gate.out].pushValue(u * b + v * a + c);
                wires_[mult_gate.out].pushTag(u * v*key_sh + u * tag_b + v * tag_a + tag_c);
            }
        }
    }

    void OnlineEvaluator::eqzEvaluate(const std::vector<common::utils::FIn1Gate> &eqz_gates) {
        if (id_ == 0) { return; }
        int pKing = 1; // Designated king party
        size_t num_eqz_gates = eqz_gates.size();
        if (num_eqz_gates == 0) { return; }
        
        std::vector<AuthAddShare> r1_send(num_eqz_gates);
        std::vector<AuthAddShare> r2_send(num_eqz_gates);

        // Compute share of m1 = input + random_value r1
        #pragma omp parallel for
        for (size_t i = 0; i < num_eqz_gates; ++i) {
            auto &eqz_gate = eqz_gates[i];
            auto *pre_eqz = static_cast<PreprocEqzGate *>(preproc_.gates[eqz_gate.out].get());
            AuthAddShare share_m1 = wires_[eqz_gate.in] + pre_eqz->share_r1;
            r1_send[i] = share_m1;
        }

        // Reconstruct the masked input m1
        std::vector<Field> recon_m1(num_eqz_gates);
        reconstruct(nP_, id_, network_, r1_send, recon_m1, check, use_pking_, latency_);

        // Compute hamming distance between bits of m1 and bits of r1
        std::vector<AuthAddShare> share_m2(num_eqz_gates);
        #pragma omp parallel for
        for (int i = 0; i < num_eqz_gates; ++i) {
            auto *pre_eqz = static_cast<PreprocEqzGate*>(preproc_.gates[eqz_gates[i].out].get());
            uint64_t m1_uint = conv<uint64_t>(recon_m1[i]);
            std::vector<Field> m1_bits(RINGSIZEBITS);
            auto num_bits = sizeof(m1_uint) * 8;
            for (size_t j = 0; j < num_bits; ++j) {
                m1_bits[j] = Field((m1_uint >> j) & 1ULL);
            }
            std::vector<AuthAddShare> r1_bits(RINGSIZEBITS);
            for (int j = 0; j < RINGSIZEBITS; ++j) {
                r1_bits[j] = pre_eqz->share_r1_bits[j];
            }
            if (id_ == 1) {
                Field ham_val = Field(0);
                Field ham_tag = Field(0);

                for (int j = 0; j < RINGSIZEBITS; ++j) {
                    ham_val += m1_bits[j] + r1_bits[j].valueAt() - 2 * m1_bits[j] * r1_bits[j].valueAt();
                    ham_tag += m1_bits[j] * r1_bits[j].keySh() + r1_bits[j].tagAt() - 2 * m1_bits[j] * r1_bits[j].tagAt();
                }

                ham_val = ham_val + pre_eqz->share_r2.valueAt();
                ham_tag = ham_tag + pre_eqz->share_r2.tagAt();
                share_m2[i].pushValue(ham_val);
                share_m2[i].pushTag(ham_tag);
            }
            else{
                Field ham_val = Field(0);
                Field ham_tag = Field(0);

                for (int j = 0; j < RINGSIZEBITS; ++j) {
                    ham_val += r1_bits[j].valueAt() - 2 * m1_bits[j] * r1_bits[j].valueAt();
                    ham_tag += m1_bits[j] * r1_bits[j].keySh() + r1_bits[j].tagAt() - 2 * m1_bits[j] * r1_bits[j].tagAt();
                }

                ham_val = ham_val + pre_eqz->share_r2.valueAt();
                ham_tag = ham_tag + pre_eqz->share_r2.tagAt();
                share_m2[i].pushValue(ham_val);
                share_m2[i].pushTag(ham_tag);
            }
        }


        // Reconstruct the masked input m2
        std::vector<Field> recon_m2(num_eqz_gates);
        reconstruct(nP_, id_, network_, share_m2, recon_m2, check, use_pking_, latency_);
 

        // Compute final output share
        #pragma omp parallel for
        for (size_t i = 0; i < num_eqz_gates; ++i) {
            auto *pre_eqz = static_cast<PreprocEqzGate *>(preproc_.gates[eqz_gates[i].out].get());
            uint64_t idx = conv<uint64_t>(recon_m2[i]) % RINGSIZEBITS;
            wires_[eqz_gates[i].out] = pre_eqz->share_r2_bits[idx];
        }
    }

    void OnlineEvaluator::recEvaluate(const std::vector<common::utils::FIn1Gate> &rec_gates) {
        if (id_ == 0) { return; }
        size_t num_rec_gates = rec_gates.size();
        std::vector<AuthAddShare> shares_to_send(num_rec_gates);

        // Gather shares to reconstruct
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            auto &rec_gate = rec_gates[i];
            shares_to_send[i] = wires_[rec_gate.in];
        }

        // Reconstruct the values
        std::vector<Field> reconstructed(num_rec_gates);
        
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, check, use_pking_, latency_);

        // Store reconstructed values in output wires
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            AuthAddShare reconstructed_share;
            reconstructed_share.pushValue(reconstructed[i]);
            // tag is not used in reconstruction, so we set it to zero
            reconstructed_share.pushTag(Field(0));
            wires_[rec_gates[i].out] = reconstructed_share;
        }
    }

    void OnlineEvaluator::rewireEvaluate(const std::vector<common::utils::SIMDOGate> &rewire_gates) {
        if (id_ == 0) { return; }
        if (rewire_gates.empty()) { return; }

        size_t num_gates = rewire_gates.size();
        
        // Extract metadata and prepare data structures for all gates in parallel
        std::vector<size_t> vec_sizes(num_gates);
        std::vector<size_t> num_payloads_vec(num_gates);
        std::vector<std::vector<AuthAddShare>> all_position_maps(num_gates);
        std::vector<std::vector<std::vector<AuthAddShare>>> all_payload_shares(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            const auto& rewire_gate = rewire_gates[g];
            
            // Input format: [pos_map_0, ..., pos_map_n, p1_0, ..., p1_n, p2_0, ..., p2_n, ...]
            // Output format: [p1_out_0, ..., p1_out_n, p2_out_0, ..., p2_out_n, ...]
            // Note: position_map wires already hold reconstructed values (public)
            
            // Get vec_size and num_payloads from preprocessing
            auto* preproc_rewire = static_cast<PreprocRewireGate*>(preproc_.gates[rewire_gate.out].get());
            size_t vec_size = preproc_rewire->vec_size;
            size_t num_payloads = preproc_rewire->num_payloads;
            
            vec_sizes[g] = vec_size;
            num_payloads_vec[g] = num_payloads;
            
            // Extract position_map values (already reconstructed/public)
            all_position_maps[g].resize(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                all_position_maps[g][i] = wires_[rewire_gate.in[i]];
            }
            
            // Extract payload shares
            all_payload_shares[g].resize(num_payloads);
            for (size_t p = 0; p < num_payloads; ++p) {
                all_payload_shares[g][p].resize(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    all_payload_shares[g][p][i] = wires_[rewire_gate.in[vec_size * (p + 1) + i]];
                }
            }
        }
        
        // Apply permutations for all gates in parallel
        std::vector<std::vector<std::unordered_map<common::utils::wire_t, AuthAddShare>>> all_temp_outputs(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            const auto& rewire_gate = rewire_gates[g];
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];
            
            all_temp_outputs[g].resize(num_payloads);
            
            // Apply public permutation based on position_map
            // For each position i: if position_map[i] = idx_perm, then output[idx_perm] = payload[i]
            for (size_t i = 0; i < vec_size; ++i) {
                // Position map is a reconstructed value (public)
                uint64_t idx_perm_uint = conv<uint64_t>(all_position_maps[g][i].valueAt());
                int idx_perm = static_cast<int>(idx_perm_uint);
                if (idx_perm >= 0 && static_cast<size_t>(idx_perm) < vec_size) {
                    for (size_t p = 0; p < num_payloads; ++p) {
                        all_temp_outputs[g][p][rewire_gate.outs[vec_size * p + idx_perm]] = all_payload_shares[g][p][i];
                    }
                }
            }
        }
        
        // Write outputs to wires (sequential to avoid race conditions)
        for (size_t g = 0; g < num_gates; ++g) {
            size_t num_payloads = num_payloads_vec[g];
            for (size_t p = 0; p < num_payloads; ++p) {
                for (const auto& [wire_id, value] : all_temp_outputs[g][p]) {
                    wires_[wire_id] = value;
                }
            }
        }
    }

    void OnlineEvaluator::permAndShEvaluate(const std::vector<common::utils::SIMDOGate> &permAndSh_gates) {
        if (id_ == 0 || permAndSh_gates.empty()) { return; }

        // Group gates by owner and batch masked shares for reconstruction
        std::vector<std::vector<AuthAddShare>> masked_shares_by_owner(nP_ + 1);
        std::vector<std::vector<AuthAddShare>> masked_tags_by_owner(nP_ + 1);
        std::vector<std::vector<size_t>> gate_indices_by_owner(nP_ + 1);
        std::vector<std::vector<size_t>> gate_offsets_by_owner(nP_ + 1);
        std::vector<size_t> gate_vec_sizes;
        
        gate_vec_sizes.reserve(permAndSh_gates.size());
        
        // Collect masked shares grouped by owner
        for (size_t gidx = 0; gidx < permAndSh_gates.size(); ++gidx) {
            const auto &gate = permAndSh_gates[gidx];
            auto *preproc = static_cast<PreprocPermAndShGate*>(preproc_.gates[gate.out].get());
            size_t vec_size = preproc->vec_size;
            int owner = preproc->owner;
            
            gate_vec_sizes.push_back(vec_size);
            gate_indices_by_owner[owner].push_back(gidx);
            gate_offsets_by_owner[owner].push_back(masked_shares_by_owner[owner].size());
            
            // Compute masked shares: T + R
            for (size_t i = 0; i < vec_size; ++i) {
                masked_shares_by_owner[owner].push_back(wires_[gate.in[i]] + preproc->mask_R[i]);
                masked_tags_by_owner[owner].push_back(wires_[gate.in[i]] + preproc->mask_R_tag[i]);
            }
        }
        
        // Batch all masked shares together for a single reconstruction call
        std::vector<AuthAddShare> all_masked_shares;
        std::vector<AuthAddShare> all_masked_tags;
        std::vector<size_t> owner_offsets(nP_ + 1, 0);
        std::vector<size_t> owner_sizes(nP_ + 1, 0);
        
        for (int owner = 1; owner <= nP_; ++owner) {
            owner_offsets[owner] = all_masked_shares.size();
            owner_sizes[owner] = masked_shares_by_owner[owner].size();
            all_masked_shares.insert(all_masked_shares.end(), 
                                     masked_shares_by_owner[owner].begin(), 
                                     masked_shares_by_owner[owner].end());
            all_masked_tags.insert(all_masked_tags.end(),
                                   masked_tags_by_owner[owner].begin(),
                                   masked_tags_by_owner[owner].end());
        }
        
        // Single reconstruction call for all shares
        std::vector<Field> all_reconstructed;
        std::vector<Field> all_reconstructed_tags;
        if (!all_masked_shares.empty()) {
            reconstruct(nP_, id_, network_, all_masked_shares, all_reconstructed, check, use_pking_, latency_, &all_masked_tags, &all_reconstructed_tags);
        }
        
        // Split reconstructed values back by owner
        std::vector<std::vector<Field>> Tprime_by_owner(nP_ + 1);
        std::vector<std::vector<Field>> Tprime_tags_by_owner(nP_ + 1);
        for (int owner = 1; owner <= nP_; ++owner) {
            if (owner_sizes[owner] > 0) {
                Tprime_by_owner[owner].assign(all_reconstructed.begin() + owner_offsets[owner],
                                              all_reconstructed.begin() + owner_offsets[owner] + owner_sizes[owner]);
                Tprime_tags_by_owner[owner].assign(all_reconstructed_tags.begin() + owner_offsets[owner],
                                                   all_reconstructed_tags.begin() + owner_offsets[owner] + owner_sizes[owner]);
            }
        }
        
        // Compute outputs for each gate
        for (int owner = 1; owner <= nP_; ++owner) {
            for (size_t idx = 0; idx < gate_indices_by_owner[owner].size(); ++idx) {
                size_t gidx = gate_indices_by_owner[owner][idx];
                const auto &gate = permAndSh_gates[gidx];
                auto *preproc = static_cast<PreprocPermAndShGate*>(preproc_.gates[gate.out].get());
                size_t vec_size = gate_vec_sizes[gidx];
                size_t offset = gate_offsets_by_owner[owner][idx];
                const std::vector<int> &owner_pi = gate.permutation[0];
                
                if (id_ == owner) {
                    // Owner computes π(T') - π(R) for their permutation
                    for (size_t i = 0; i < vec_size; ++i) {
                        int idx_perm = owner_pi[i];
                        wires_[gate.outs[i]].pushValue(Tprime_by_owner[owner][offset + idx_perm] - preproc->permuted_mask[i].valueAt());
                        wires_[gate.outs[i]].pushTag(Tprime_tags_by_owner[owner][offset + idx_perm] - preproc->permuted_mask_tag[i].tagAt());
                    }
                } else {
                    // Non-owners compute -π_i(R)
                    for (size_t i = 0; i < vec_size; ++i) {
                        wires_[gate.outs[i]].pushValue(-preproc->permuted_mask[i].valueAt());
                        wires_[gate.outs[i]].pushTag(-preproc->permuted_mask_tag[i].tagAt());
                    }
                }
            }
        }
    }

    void OnlineEvaluator::cPermAndShEvaluate(const std::vector<common::utils::SIMDOGate> &cPermAndSh_gates) {
        if (id_ == 0 || cPermAndSh_gates.empty()) { return; }

        // Group gates by owner and batch masked shares for reconstruction
        std::vector<std::vector<AuthAddShare>> masked_shares_by_owner(nP_ + 1);
        std::vector<std::vector<AuthAddShare>> masked_tags_by_owner(nP_ + 1);
        std::vector<std::vector<size_t>> gate_indices_by_owner(nP_ + 1);
        std::vector<std::vector<size_t>> gate_offsets_by_owner(nP_ + 1);
        std::vector<size_t> gate_vec_sizes;
        
        gate_vec_sizes.reserve(cPermAndSh_gates.size());
        
        // Collect masked shares grouped by owner
        for (size_t gidx = 0; gidx < cPermAndSh_gates.size(); ++gidx) {
            const auto &gate = cPermAndSh_gates[gidx];
            auto *preproc = static_cast<PreprocCPermAndShGate*>(preproc_.gates[gate.out].get());
            size_t vec_size = preproc->vec_size;
            int owner = preproc->owner;
            
            gate_vec_sizes.push_back(vec_size);
            gate_indices_by_owner[owner].push_back(gidx);
            gate_offsets_by_owner[owner].push_back(masked_shares_by_owner[owner].size());
            
            // Input structure: [input..., add_to_input..., sub_from_output...]
            // Compute masked shares: (T + add_to_input) + R
            for (size_t i = 0; i < vec_size; ++i) {
                AuthAddShare masked_wire;
                masked_shares_by_owner[owner].push_back(wires_[gate.in[i]] + wires_[gate.in[vec_size + i]] + preproc->mask_R[i]);
                masked_tags_by_owner[owner].push_back(wires_[gate.in[i]] + preproc->mask_R_tag[i]);
            }
        }
        
        // Batch all masked shares together for a single reconstruction call
        std::vector<AuthAddShare> all_masked_shares;
        std::vector<AuthAddShare> all_masked_tags;
        std::vector<size_t> owner_offsets(nP_ + 1, 0);
        std::vector<size_t> owner_sizes(nP_ + 1, 0);
        
        for (int owner = 1; owner <= nP_; ++owner) {
            owner_offsets[owner] = all_masked_shares.size();
            owner_sizes[owner] = masked_shares_by_owner[owner].size();
            all_masked_shares.insert(all_masked_shares.end(), 
                                     masked_shares_by_owner[owner].begin(), 
                                     masked_shares_by_owner[owner].end());
            all_masked_tags.insert(all_masked_tags.end(),
                                   masked_tags_by_owner[owner].begin(),
                                   masked_tags_by_owner[owner].end());
        }
        
        // Single reconstruction call for all shares
        std::vector<Field> all_reconstructed;
        std::vector<Field> all_reconstructed_tags;
        if (!all_masked_shares.empty()) {
            reconstruct(nP_, id_, network_, all_masked_shares, all_reconstructed, check, use_pking_, latency_, &all_masked_tags, &all_reconstructed_tags);
        }
        
        // Split reconstructed values back by owner
        std::vector<std::vector<Field>> Tprime_by_owner(nP_ + 1);
        std::vector<std::vector<Field>> Tprime_tags_by_owner(nP_ + 1);
        for (int owner = 1; owner <= nP_; ++owner) {
            if (owner_sizes[owner] > 0) {
                Tprime_by_owner[owner].assign(all_reconstructed.begin() + owner_offsets[owner],
                                              all_reconstructed.begin() + owner_offsets[owner] + owner_sizes[owner]);
                Tprime_tags_by_owner[owner].assign(all_reconstructed_tags.begin() + owner_offsets[owner],
                                                   all_reconstructed_tags.begin() + owner_offsets[owner] + owner_sizes[owner]);
            }
        }
        
        // Compute outputs for each gate
        for (int owner = 1; owner <= nP_; ++owner) {
            for (size_t idx = 0; idx < gate_indices_by_owner[owner].size(); ++idx) {
                size_t gidx = gate_indices_by_owner[owner][idx];
                const auto &gate = cPermAndSh_gates[gidx];
                auto *preproc = static_cast<PreprocCPermAndShGate*>(preproc_.gates[gate.out].get());
                size_t vec_size = gate_vec_sizes[gidx];
                size_t offset = gate_offsets_by_owner[owner][idx];
                const std::vector<int> &owner_pi = gate.permutation[0];
                
                if (id_ == owner) {
                    // Owner computes π(T') - π(R) - sub_from_output for their permutation
                    for (size_t i = 0; i < vec_size; ++i) {
                        int idx_perm = owner_pi[i];
                        wires_[gate.outs[i]].pushValue(Tprime_by_owner[owner][offset + idx_perm] - preproc->permuted_mask[i].valueAt() - wires_[gate.in[2 * vec_size + i]].valueAt());
                        wires_[gate.outs[i]].pushTag(Tprime_tags_by_owner[owner][offset + idx_perm] - preproc->permuted_mask_tag[i].tagAt());
                    }
                } else {
                    // Non-owners compute -π_i(R) - sub_from_output
                    for (size_t i = 0; i < vec_size; ++i) {
                        wires_[gate.outs[i]].pushValue(-preproc->permuted_mask[i].valueAt() - wires_[gate.in[2 * vec_size + i]].valueAt());
                        wires_[gate.outs[i]].pushTag(-preproc->permuted_mask_tag[i].tagAt());
                    }
                }
            }
        }
    }

    void OnlineEvaluator::amortzdPnSEvaluate(const std::vector<common::utils::SIMDOGate> &amortzdPnS_gates) {
        if (id_ == 0 || amortzdPnS_gates.empty()) { return; }

        // Batch mask-and-reconstruct across all amortzdPnS gates.
        std::vector<AuthAddShare> all_masked_shares;
        std::vector<AuthAddShare> all_masked_shares_tags;
        std::vector<size_t> vec_sizes;
        std::vector<const common::utils::SIMDOGate*> gates_order;

        for (const auto &gate : amortzdPnS_gates) {
            auto *preproc = static_cast<PreprocAmortzdPnSGate *>(preproc_.gates[gate.outs[0]].get());
            size_t vec_size = preproc->vec_size;
            vec_sizes.push_back(vec_size);
            gates_order.push_back(&gate);
            for (size_t i = 0; i < vec_size; i++) {
                all_masked_shares.push_back(wires_[gate.in[i]] + preproc->mask_R[i]);
                all_masked_shares_tags.push_back(wires_[gate.in[i]] + preproc->mask_R_tag[i]);
            }
        }

        if (all_masked_shares.empty()) { return; }

        std::vector<Field> all_reconstructed;
        std::vector<Field> all_reconstructed_tags;
        
        reconstruct(nP_, id_, network_, all_masked_shares, all_reconstructed, check, use_pking_, latency_, &all_masked_shares_tags, &all_reconstructed_tags);


        // Now split and compute outputs per gate. For each amortzdPnS gate, we need to produce nP * vec_size outputs (flattened).
        size_t offset = 0;
        for (size_t gidx = 0; gidx < amortzdPnS_gates.size(); ++gidx) {
            const auto *gate = gates_order[gidx];
            auto *preproc = static_cast<PreprocAmortzdPnSGate *>(preproc_.gates[gate->outs[0]].get());
            size_t vec_size = vec_sizes[gidx];
            size_t nP = preproc->nP;

            // T' for this gate is all_Tprime[offset ... offset+vec_size-1]
            // Fill outputs in flattened order: for party 0..nP-1, each has vec_size outputs
            size_t out_idx = 0;
            for (size_t party_idx = 0; party_idx < nP; party_idx++) {
                if (party_idx + 1 == id_) {
                    const std::vector<int>& my_permutation = gate->permutation[party_idx];
                    for (size_t j = 0; j < vec_size; j++) {
                        int idx_perm = my_permutation[j];
                        wires_[gate->outs[out_idx]].pushValue(all_reconstructed[offset + idx_perm] - preproc->permuted_masks[party_idx][j].valueAt());
                        wires_[gate->outs[out_idx]].pushTag(all_reconstructed_tags[offset + idx_perm] - preproc->permuted_masks_tag[party_idx][j].tagAt());
                        out_idx++;
                    }
                } else {
                    for (size_t j = 0; j < vec_size; j++) {
                        wires_[gate->outs[out_idx]].pushValue(-preproc->permuted_masks[party_idx][j].valueAt());
                        wires_[gate->outs[out_idx]].pushTag(-preproc->permuted_masks_tag[party_idx][j].tagAt());
                        out_idx++;
                    }
                }
            }

            offset += vec_size;
        }
    }

    void OnlineEvaluator::cAmortzdPnSEvaluate(const std::vector<common::utils::SIMDOGate> &cAmortzdPnS_gates) {
        if (id_ == 0 || cAmortzdPnS_gates.empty()) { return; }

        // Batch mask-and-reconstruct across all cAmortzdPnS gates.
        // cAmortzdPnS: input = [input..., commitment..., perm_comm0..., perm_comm1..., ..., perm_commN-1...]
        // Logic: mask (input + commitment) and reconstruct
        std::vector<AuthAddShare> all_masked_shares;
        std::vector<AuthAddShare> all_masked_shares_tags;
        std::vector<size_t> vec_sizes;
        std::vector<const common::utils::SIMDOGate*> gates_order;

        for (const auto &gate : cAmortzdPnS_gates) {
            auto *preproc = static_cast<PreprocCAmortzdPnSGate *>(preproc_.gates[gate.outs[0]].get());
            size_t vec_size = preproc->vec_size;
            vec_sizes.push_back(vec_size);
            gates_order.push_back(&gate);
            
            // Add commitment to input: (input[i] + commitment[i]) + mask_R[i]
            // Input layout: [input..., commitment..., perm_comm0..., perm_comm1..., ...]
            // Note: tags only include input, not commitment (following cPermAndSh pattern)
            for (size_t i = 0; i < vec_size; ++i) {
                all_masked_shares.push_back(wires_[gate.in[i]] + wires_[gate.in[vec_size + i]] + preproc->mask_R[i]);
                all_masked_shares_tags.push_back(wires_[gate.in[i]] + preproc->mask_R_tag[i]);
            }
        }

        if (all_masked_shares.empty()) { return; }

        std::vector<Field> all_reconstructed;
        std::vector<Field> all_reconstructed_tags;
        
        reconstruct(nP_, id_, network_, all_masked_shares, all_reconstructed, check, use_pking_, latency_, &all_masked_shares_tags, &all_reconstructed_tags);

        // Now split and compute outputs per gate. For each cAmortzdPnS gate, produce nP * vec_size outputs (flattened).
        // Output: pi(input + commitment) - permuted_commitment_i for party i
        size_t offset = 0;
        for (size_t gidx = 0; gidx < cAmortzdPnS_gates.size(); ++gidx) {
            const auto *gate = gates_order[gidx];
            auto *preproc = static_cast<PreprocCAmortzdPnSGate *>(preproc_.gates[gate->outs[0]].get());
            size_t vec_size = vec_sizes[gidx];
            size_t nP = preproc->nP;

            // T' for this gate is all_reconstructed[offset ... offset+vec_size-1]
            // Fill outputs in flattened order: for party 0..nP-1, each has vec_size outputs
            size_t out_idx = 0;
            for (size_t party_idx = 0; party_idx < nP; party_idx++) {
                if (party_idx + 1 == id_) {
                    const std::vector<int>& my_permutation = gate->permutation[party_idx];
                    for (size_t j = 0; j < vec_size; ++j) {
                        int idx_perm = my_permutation[j];
                        // Output = pi(T') - pi(R) - permuted_commitment[party_idx][j]
                        // permuted_commitment is at gate.in[2*vec_size + party_idx*vec_size + j]
                        size_t perm_comm_idx = (2 + party_idx) * vec_size + j;
                        wires_[gate->outs[out_idx]].pushValue(all_reconstructed[offset + idx_perm] 
                                                               - preproc->permuted_masks[party_idx][j].valueAt()
                                                               - wires_[gate->in[perm_comm_idx]].valueAt());
                        wires_[gate->outs[out_idx]].pushTag(all_reconstructed_tags[offset + idx_perm] 
                                                             - preproc->permuted_masks_tag[party_idx][j].tagAt());
                        out_idx++;
                    }
                } else {
                    for (size_t j = 0; j < vec_size; ++j) {
                        // Output = -pi(R) - permuted_commitment[party_idx][j]
                        size_t perm_comm_idx = (2 + party_idx) * vec_size + j;
                        wires_[gate->outs[out_idx]].pushValue(Field(0) 
                                                               - preproc->permuted_masks[party_idx][j].valueAt()
                                                               - wires_[gate->in[perm_comm_idx]].valueAt());
                        wires_[gate->outs[out_idx]].pushTag(Field(0) 
                                                             - preproc->permuted_masks_tag[party_idx][j].tagAt());
                        out_idx++;
                    }
                }
            }

            offset += vec_size;
        }
    }

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }

        std::vector<common::utils::FIn2Gate> mult_gates;
        std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::FIn1Gate> rec_gates;
        // std::vector<common::utils::SIMDOGate> shuffle_gates;
        std::vector<common::utils::SIMDOGate> permAndSh_gates;
        std::vector<common::utils::SIMDOGate> cPermAndSh_gates;
        std::vector<common::utils::SIMDOGate> amortzdPnS_gates;
        std::vector<common::utils::SIMDOGate> cAmortzdPnS_gates;
        std::vector<common::utils::SIMDOGate> rewire_gates;

        // First pass: collect the multi-round gates so their batched handlers can run once.
        for (auto &gate : circ_.gates_by_level[depth]) {
            switch (gate->type) {
                case common::utils::GateType::kMul: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    mult_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kEqz: {
                    auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                    eqz_gates.push_back(*g);
                    break;
                }
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
                
                case common::utils::GateType::kPermAndSh: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    permAndSh_gates.push_back(*g);
                    break;
                }

                case common::utils::GateType::kCPermAndSh: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    cPermAndSh_gates.push_back(*g);
                    break;
                }

                case common::utils::GateType::kAmortzdPnS: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    amortzdPnS_gates.push_back(*g);
                    break;
                }

                case common::utils::GateType::kCAmortzdPnS: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    cAmortzdPnS_gates.push_back(*g);
                    break;
                }

                case common::utils::GateType::kRewire: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Rewire gates are processed immediately
                    rewire_gates.push_back(*g);
                    break;
                }

                default:
                    break;
            }
        }

        if (!mult_gates.empty()) { multEvaluate(mult_gates); }
        if (!eqz_gates.empty()) { eqzEvaluate(eqz_gates); }
        if (!rec_gates.empty()) { recEvaluate(rec_gates); }
        // if (!shuffle_gates.empty()) {  }
        if (!permAndSh_gates.empty()) { permAndShEvaluate(permAndSh_gates); }
        if (!cPermAndSh_gates.empty()) { cPermAndShEvaluate(cPermAndSh_gates); }
        if (!amortzdPnS_gates.empty()) { amortzdPnSEvaluate(amortzdPnS_gates); }
        if (!cAmortzdPnS_gates.empty()) { cAmortzdPnSEvaluate(cAmortzdPnS_gates); }
        if (!rewire_gates.empty()) { rewireEvaluate(rewire_gates); }
        
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
            // Gather output shares as pairs (value, tag)
            std::vector<AuthAddShare> output_shares(circ_.outputs.size());
            for (size_t i = 0; i < circ_.outputs.size(); ++i) {
                auto wout = circ_.outputs[i];
                output_shares[i] = wires_[wout];
            }
            
            // Use reconstruct function to get the output values
            reconstruct(nP_, id_, network_, output_shares, outvals, check, use_pking_, latency_);
        }
        
        verify(check);

        return outvals;
    }

    std::vector<Field> OnlineEvaluator::evaluateCircuit(const std::unordered_map<common::utils::wire_t, Field> &inputs) {
        setInputs(inputs);
        for (size_t i = 0; i < circ_.gates_by_level.size(); ++i) {
            evaluateGatesAtDepth(i);
        }
        return getOutputs();
    }

    void OnlineEvaluator::verify(AuthAddShare check_val) {
        if (id_ == 0) { return; }

        // Compute commitment on check value and tag
        // placeholder for now
        Field commitment = check_val.valueAt() + check_val.tagAt();
        
        // Exchange commitment values with all parties
        std::vector<Field> commitment_values(nP_);
        commitment_values[id_ - 1] = commitment;
        
        // Send commitment value to all other parties
        for (int p = 1; p <= nP_; ++p) {
            if (p != id_) {
                network_->send(p, &commitment, sizeof(Field));
            }
        }
        
        usleep(latency_);
        
        // Receive commitment values from all other parties
        for (int p = 1; p <= nP_; ++p) {
            if (p != id_) {
                network_->recv(p, &commitment_values[p - 1], sizeof(Field));
            }
        }
        
        // Step 3: Exchange value and tag shares
        std::vector<Field> value_shares(nP_);
        std::vector<Field> tag_shares(nP_);
        value_shares[id_ - 1] = check_val.valueAt();
        tag_shares[id_ - 1] = check_val.tagAt();
        
        // Send value and tag shares to all other parties
        for (int p = 1; p <= nP_; ++p) {
            if (p != id_) {
                Field shares[2] = {check_val.valueAt(), check_val.tagAt()};
                network_->send(p, shares, 2 * sizeof(Field));
            }
        }
        
        usleep(latency_);
        
        // Receive value and tag shares from all other parties
        for (int p = 1; p <= nP_; ++p) {
            if (p != id_) {
                Field shares[2];
                network_->recv(p, shares, 2 * sizeof(Field));
                value_shares[p - 1] = shares[0];
                tag_shares[p - 1] = shares[1];
            }
        }
        
        // Step 4: Verify Commitment matches
        bool commitment_match = true;
        for (int p = 0; p < nP_; ++p) {
            Field expected_commitment = value_shares[p] + tag_shares[p];
            if (commitment_values[p] != expected_commitment) {
                commitment_match = false;
                break;
            }
        }
        
        // Step 5: Reconstruct and check if value and tag sum to zero
        Field reconstructed_value = Field(0);
        Field reconstructed_tag = Field(0);
        
        for (int p = 0; p < nP_; ++p) {
            reconstructed_value += value_shares[p];
            reconstructed_tag += tag_shares[p];
        }
        
        // Step 6: Check verification conditions
        bool value_zero = (reconstructed_value == Field(0));
        bool tag_zero = (reconstructed_tag == Field(0));
        
        if (commitment_match && value_zero && tag_zero) {

                std::cout << "Verification of all reconstruction passed" << std::endl;

        } else {

                std::cout << "Verification of all reconstruction failed";
                if (!commitment_match) std::cout << " (Commitment mismatch)";
                if (!value_zero) std::cout << " (value != 0)";
                if (!tag_zero) std::cout << " (tag != 0)";
                std::cout << std::endl;

        }
    }

}; // namespace graphdb