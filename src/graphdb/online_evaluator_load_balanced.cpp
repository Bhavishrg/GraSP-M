#include "online_evaluator.h"

#include "../utils/helpers.h"

namespace graphdb
{
    // Helper function to reconstruct shares via king party or direct all-to-all
    void OnlineEvaluator::reconstruct(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                                     const std::vector<Ring>& shares_list,
                                     std::vector<Ring>& reconstructed_list,
                                     bool via_pking, int latency) {
        int pKing = 1;
        size_t num_shares = shares_list.size();
        reconstructed_list.resize(num_shares, 0);
        
        if (via_pking) {
            // Reconstruction via king party
            if (pid != pKing) {
                network->send(pKing, shares_list.data(), shares_list.size() * sizeof(Ring));
                usleep(latency);
                network->recv(pKing, reconstructed_list.data(), reconstructed_list.size() * sizeof(Ring));
            } else {
                std::vector<std::vector<Ring>> share_recv(nP);
                share_recv[pKing - 1] = shares_list;
                usleep(latency);
                
                // Receive from all parties (not parallelized as recv is blocking)
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        share_recv[p - 1].resize(num_shares);
                        network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(Ring));
                    }
                }
                
                // Aggregate shares
                for (int p = 0; p < nP; ++p) {
                    for (size_t i = 0; i < num_shares; ++i) {
                        reconstructed_list[i] += share_recv[p][i];
                    }
                }
                
                // Send result to all parties (sequential for now to avoid race conditions)
                for (int p = 1; p <= nP; ++p) {
                    if (p != pKing) {
                        network->send(p, reconstructed_list.data(), reconstructed_list.size() * sizeof(Ring));
                        network->flush(p);
                    }
                }
            }
        } else {
            // Direct reconstruction (all parties exchange shares)
            std::vector<std::vector<Ring>> share_recv(nP);
            share_recv[pid - 1] = shares_list;
            
            // Send to all parties (sequential for now to avoid race conditions)
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    network->send(p, shares_list.data(), shares_list.size() * sizeof(Ring));
                }
            }
            
            usleep(latency);
            
            // Receive from all parties
            for (int p = 1; p <= nP; ++p) {
                if (p != pid) {
                    share_recv[p - 1].resize(num_shares);
                    network->recv(p, share_recv[p - 1].data(), share_recv[p - 1].size() * sizeof(Ring));
                }
            }
            
            // Aggregate shares
            for (int p = 0; p < nP; ++p) {
                for (size_t i = 0; i < num_shares; ++i) {
                    reconstructed_list[i] += share_recv[p][i];
                }
            }
        }
    }

    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit<Ring> preproc,
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
                                     PreprocCircuit<Ring> preproc,
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

    
          void OnlineEvaluator::setInputs(const std::unordered_map<common::utils::wire_t, Ring> &inputs) {
        // Input gates have depth 0
        for (auto &g : circ_.gates_by_level[0]) {
            if (g->type == common::utils::GateType::kInp) {
                auto *pre_input = static_cast<PreprocInput<Ring> *>(preproc_.gates[g->out].get());
                auto pid = pre_input->pid;
                if (id_ != 0) {
                    if (pid == id_) {
                        Ring accumulated_val = Ring(0);
                        for (size_t i = 1; i <= nP_; i++) {
                            if (i != pid) {
                                Ring rand_sh;
                                rgen_.pi(i).random_data(&rand_sh, sizeof(Ring));
                                accumulated_val += rand_sh;
                            }
                        }
                        wires_[g->out] = inputs.at(g->out) - accumulated_val;
                    } else {
                        rgen_.pi(id_).random_data(&wires_[g->out], sizeof(Ring));
                    }
                }
            }
        }
    }

    void OnlineEvaluator::setRandomInputs() {
        // Input gates have depth 0.
        for (auto &g : circ_.gates_by_level[0]) {
            if (g->type == common::utils::GateType::kInp) {
                rgen_.pi(id_).random_data(&wires_[g->out], sizeof(Ring));
            }
        }
    }

    void OnlineEvaluator::eqzEvaluate(const std::vector<common::utils::FIn1Gate> &eqz_gates) {
        if (id_ == 0) { return; }
        int pKing = 1; // Designated king party
        size_t num_eqz_gates = eqz_gates.size();
        std::vector<Ring> r1_send(num_eqz_gates);
        std::vector<Ring> r2_send(num_eqz_gates);

        // Compute share of m1 = input + random_value r1
        #pragma omp parallel for
        for (size_t i = 0; i < num_eqz_gates; ++i) {
            auto &eqz_gate = eqz_gates[i];
            auto *pre_eqz = static_cast<PreprocEqzGate<Ring> *>(preproc_.gates[eqz_gate.out].get());
            Ring share_m1 = wires_[eqz_gate.in] + pre_eqz->share_r1.valueAt();
            r1_send[i] = share_m1;
        }

        // Reconstruct the masked input m1
        std::vector<Ring> recon_m1(num_eqz_gates, 0);
        reconstruct(nP_, id_, network_, r1_send, recon_m1, use_pking_, latency_);

        // Compute hamming distance between bits of m1 and bits of r1
        std::vector<Ring> share_m2(num_eqz_gates, 0);
        #pragma omp parallel for
        for (int i = 0; i < num_eqz_gates; ++i) {
            auto *pre_eqz = static_cast<PreprocEqzGate<Ring> *>(preproc_.gates[eqz_gates[i].out].get());
            std::vector<Ring> m1_bits(RINGSIZEBITS);
            m1_bits = bitDecomposeToInt(recon_m1[i]);
            std::vector<Ring> r1_bits(RINGSIZEBITS);
            for (int j = 0; j < RINGSIZEBITS; ++j) {
                r1_bits[j] = pre_eqz->share_r1_bits[j].valueAt();
            }
            if (id_ == 1) {
                for (int j = 0; j < RINGSIZEBITS; ++j) {
                    share_m2[i] += m1_bits[j] + r1_bits[j] - 2 * m1_bits[j] * r1_bits[j];
                }
                share_m2[i] += pre_eqz->share_r2.valueAt();
            }
            else{
                for (int j = 0; j < RINGSIZEBITS; ++j) {
                    share_m2[i] += r1_bits[j] - 2 * m1_bits[j] * r1_bits[j];
                }
                share_m2[i] += pre_eqz->share_r2.valueAt();
            }
        }


        // Reconstruct the masked input m2
        std::vector<Ring> recon_m2(num_eqz_gates, 0);
        reconstruct(nP_, id_, network_, share_m2, recon_m2, use_pking_, latency_);
 

        // Compute final output share
        std::vector<Ring> recon_out(num_eqz_gates, 0);
        #pragma omp parallel for
        for (int i = 0; i < num_eqz_gates; ++i) {
            auto *pre_eqz = static_cast<PreprocEqzGate<Ring> *>(preproc_.gates[eqz_gates[i].out].get());
            std::vector<Ring> r2_bits(RINGSIZEBITS);
            for (int j = 0; j < RINGSIZEBITS; ++j) {
                r2_bits[j] = pre_eqz->share_r2_bits[j].valueAt();
            }
            recon_out[i] = r2_bits[recon_m2[i]%RINGSIZEBITS];
            wires_[eqz_gates[i].out] = recon_out[i]; // Reconstructed output
        }
    }

    void OnlineEvaluator::recEvaluate(const std::vector<common::utils::FIn1Gate> &rec_gates) {
        if (id_ == 0) { return; }
        size_t num_rec_gates = rec_gates.size();
        std::vector<Ring> shares_to_send(num_rec_gates);

        // Gather shares to reconstruct
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            auto &rec_gate = rec_gates[i];
            shares_to_send[i] = wires_[rec_gate.in];
        }

        // Reconstruct the values
        std::vector<Ring> reconstructed(num_rec_gates, 0);
        
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);

        // Store reconstructed values in output wires
        #pragma omp parallel for
        for (size_t i = 0; i < num_rec_gates; ++i) {
            wires_[rec_gates[i].out] = reconstructed[i];
        }
    }


    void OnlineEvaluator::shuffleEvaluate(const std::vector<common::utils::SIMDOGate> &shuffle_gates) {
        if (id_ == 0 || shuffle_gates.empty()) { return; }

        std::vector<Ring> z_all;
        std::vector<std::vector<Ring>> z_sum;
        size_t total_comm = 0;

        for (const auto &gate : shuffle_gates) {
            auto *pre_shuffle = static_cast<PreprocShuffleGate<Ring> *>(preproc_.gates[gate.out].get());
            size_t vec_size = gate.in.size();
            total_comm += vec_size;
            std::vector<Ring> z(vec_size, 0);
            
            for (int i = 0; i < vec_size; ++i) {
                z[i] = wires_[gate.in[i]] + pre_shuffle->a[i].valueAt();
            }
            z_sum.push_back(z);
        }
        
        if (id_ == 1) {
            // Party 1 collects z values from all parties
            std::vector<std::vector<Ring>> z_recv_all(nP_);
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_recv_all[pid - 1] = std::vector<Ring>(total_comm);
                network_->recv(pid, z_recv_all[pid - 1].data(), z_recv_all[pid - 1].size() * sizeof(Ring));
            }
            usleep(latency_);

            // Aggregate received z values from other parties
            for (int pid = 2; pid <= nP_; ++pid) {
                size_t idx_vec = 0;
                for (int idx_gate = 0; idx_gate < shuffle_gates.size(); ++idx_gate) {
                    size_t vec_size = shuffle_gates[idx_gate].in.size();
                    std::vector<Ring> z(z_recv_all[pid - 1].begin() + idx_vec, z_recv_all[pid - 1].begin() + idx_vec + vec_size);
                    for (int i = 0; i < vec_size; ++i) {
                        z_sum[idx_gate][i] += z[i];
                    }
                    idx_vec += vec_size;
                }
            }


            z_all.reserve(total_comm);

            for (int idx_gate = 0; idx_gate < shuffle_gates.size(); ++idx_gate) {
                auto *pre_shuffle = static_cast<PreprocShuffleGate<Ring> *>(preproc_.gates[shuffle_gates[idx_gate].out].get());
                size_t vec_size = shuffle_gates[idx_gate].in.size();
                std::vector<Ring> z(vec_size);
                #pragma omp parallel for
                for (int i = 0; i < vec_size; ++i) {
                    z[i] = z_sum[idx_gate][pre_shuffle->pi[i]] + pre_shuffle->b[pre_shuffle->pi[i]].valueAt();
                    wires_[shuffle_gates[idx_gate].outs[i]] = pre_shuffle->c[i].valueAt();
                }
                z_all.insert(z_all.end(), z.begin(), z.end());
            }
            network_->send(2, z_all.data(), z_all.size() * sizeof(Ring));

        } else {

            // Flatten z_sum into z_all before sending
            z_all.clear();
            z_all.reserve(total_comm);
            for (const auto& z_vec : z_sum) {
                z_all.insert(z_all.end(), z_vec.begin(), z_vec.end());
            }
            
            network_->send(1, z_all.data(), z_all.size() * sizeof(Ring));
            network_->flush(1);


            z_all.clear();
            z_all.resize(total_comm);
            network_->recv(id_ - 1, z_all.data(), z_all.size() * sizeof(Ring));
            usleep(latency_);

            for (int idx_gate = 0, idx_vec = 0; idx_gate < shuffle_gates.size(); ++idx_gate) {
                auto *pre_shuffle = static_cast<PreprocShuffleGate<Ring> *>(preproc_.gates[shuffle_gates[idx_gate].out].get());
                size_t vec_size = shuffle_gates[idx_gate].in.size();
                std::vector<Ring> z(z_all.begin() + idx_vec, z_all.begin() + idx_vec + vec_size);
                std::vector<Ring> z_send(vec_size);
                for (int i = 0; i < vec_size; ++i) {
                    if (id_ != nP_) {
                        z_send[i] = z[pre_shuffle->pi[i]] + pre_shuffle->b[pre_shuffle->pi[i]].valueAt();
                        wires_[shuffle_gates[idx_gate].outs[i]] = pre_shuffle->c[i].valueAt();
                    } else {
                        z_send[i] = z[pre_shuffle->pi[i]] + pre_shuffle->b[pre_shuffle->pi[i]].valueAt() - pre_shuffle->delta[i];
                        wires_[shuffle_gates[idx_gate].outs[i]] = z_send[i] + pre_shuffle->c[i].valueAt();
                    }
                    z_all[idx_vec++] = z_send[i];
                }
            }
            if (id_ != nP_) {
                network_->send(id_ + 1, z_all.data(), z_all.size() * sizeof(Ring));
            }
        }
    }


        
    void OnlineEvaluator::permAndShEvaluate(const std::vector<common::utils::SIMDOGate> &permAndSh_gates) {
        if (id_ == 0) { return; }
        
        // Use thread pool to execute send and receive in parallel
        auto send_future = tpool_->enqueue([&]() {
            for (int idx_gate = 0; idx_gate < permAndSh_gates.size(); ++idx_gate) {
                auto *pre_permAndSh = static_cast<PreprocPermAndShGate<Ring> *>(preproc_.gates[permAndSh_gates[idx_gate].out].get());
                size_t vec_size = permAndSh_gates[idx_gate].in.size();
                std::vector<Ring> z(vec_size, 0);
                if (id_ != permAndSh_gates[idx_gate].owner) {
                    for (int i = 0; i < vec_size; ++i) {
                        z[i] = wires_[permAndSh_gates[idx_gate].in[i]] - pre_permAndSh->a[i].valueAt();
                        wires_[permAndSh_gates[idx_gate].outs[i]] = pre_permAndSh->b[i].valueAt();
                    }
                    network_->send(permAndSh_gates[idx_gate].owner, z.data(), z.size() * sizeof(Ring));
                    network_->flush(permAndSh_gates[idx_gate].owner);
                }
            }
        });

        auto recv_future = tpool_->enqueue([&]() {
            usleep(latency_);
            for (int idx_gate = 0; idx_gate < permAndSh_gates.size(); ++idx_gate) {
                if (id_ == permAndSh_gates[idx_gate].owner) {
                    auto *pre_permAndSh = static_cast<PreprocPermAndShGate<Ring> *>(preproc_.gates[permAndSh_gates[idx_gate].out].get());
                    size_t vec_size = permAndSh_gates[idx_gate].in.size();
                    std::vector<std::vector<Ring>> z(nP_, std::vector<Ring>(vec_size, 0));
                    
                    for (int pid = 1; pid <= nP_; ++pid) {
                        if (pid != permAndSh_gates[idx_gate].owner) {
                            std::vector<Ring> z_recv(vec_size);
                            network_->recv(pid, z_recv.data(), z_recv.size() * sizeof(Ring));
                            for (int i = 0; i < vec_size; ++i) {
                                z[pid - 1][i] += z_recv[i];
                            }
                        } else {
                            for (int i = 0; i < vec_size; ++i) {
                                z[pid - 1][i] += wires_[permAndSh_gates[idx_gate].in[i]];
                            }
                        }
                    }
                    for (int i = 0; i < vec_size; ++i) {
                        Ring sum = Ring(0);
                        for (int pid = 0; pid < nP_; ++pid) {
                            sum += z[pid][pre_permAndSh->pi[i]];
                        }
                        wires_[permAndSh_gates[idx_gate].outs[i]] = sum + pre_permAndSh->delta[i].valueAt();
                    }
                }
            }
        });
        
        // Wait for both send and receive operations to complete
        if (send_future.valid()) {
            send_future.wait();
        }
        if (recv_future.valid()) {
            recv_future.wait();
        }
    }

    void OnlineEvaluator::amortzdPnSEvaluate(const std::vector<common::utils::SIMDOGate> &amortzdPnS_gates) {
        if (id_ == 0 || amortzdPnS_gates.empty()) { return; }
        
        // Protocol:
        // 1. Each party masks their input T with R: compute T + R
        // 2. Reconstruct T' = T + R (public)
        // 3. Party i computes: ⟨T_i⟩_i = π_i(T') - ⟨π_i(R)⟩_i
        // 4. Party i computes: ⟨T_j⟩_i = -⟨π_j(R)⟩_i for j ≠ i
        
        for (const auto &gate : amortzdPnS_gates) {
            auto *preproc = static_cast<PreprocAmortzdPnSGate<Ring> *>(preproc_.gates[gate.outs[0]].get());
            size_t vec_size = preproc->vec_size;
            size_t nP = preproc->nP;
            
            // Step 1: Mask input with R: compute shares of T + R
            std::vector<Ring> masked_shares(vec_size);
            for (size_t i = 0; i < vec_size; i++) {
                masked_shares[i] = wires_[gate.in[i]] + preproc->mask_R[i].valueAt();
            }
            
            // Step 2: Reconstruct T' = T + R
            std::vector<Ring> T_prime(vec_size);
            reconstruct(nP_, id_, network_, masked_shares, T_prime, use_pking_, latency_);
            
            // Step 3 & 4: Compute output shares
            // Outputs are flattened: [party0_vec, party1_vec, ..., partyNP_vec]
            size_t out_idx = 0;
            
            for (size_t party_idx = 0; party_idx < nP; party_idx++) {
                if (party_idx + 1 == id_) {
                    // This is my output: ⟨T_i⟩_i = π_i(T') - ⟨π_i(R)⟩_i
                    const std::vector<int>& my_permutation = gate.permutation[0];
                    for (size_t j = 0; j < vec_size; j++) {
                        int idx_perm = my_permutation[j];
                        wires_[gate.outs[out_idx++]] = T_prime[idx_perm] - preproc->permuted_masks[party_idx][idx_perm].valueAt();
                    }
                } else {
                    // Other party's output: ⟨T_j⟩_i = -⟨π_j(R)⟩_i
                    for (size_t j = 0; j < vec_size; j++) {
                        // We don't know the permutation, so we use shares in original order
                        // The preprocessing already computed π_j(R), so shares are already permuted
                        wires_[gate.outs[out_idx++]] = -preproc->permuted_masks[party_idx][j].valueAt();
                    }
                }
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
        
        std::vector<Ring> u_shares(num_mult_gates);
        std::vector<Ring> v_shares(num_mult_gates);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_mult_gates; ++i) {
            auto &mult_gate = mult_gates[i];
            auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[mult_gate.out].get());
            
            // Compute this party's share of u and v
            u_shares[i] = wires_[mult_gate.in1] - pre_out->triple_a.valueAt();
            v_shares[i] = wires_[mult_gate.in2] - pre_out->triple_b.valueAt();
        }
        
        // Step 2: Reconstruct u and v values
        std::vector<Ring> u_reconstructed(num_mult_gates, 0);
        std::vector<Ring> v_reconstructed(num_mult_gates, 0);
        
        // Prepare shares to send (interleaved: u0, v0, u1, v1, ...)
        std::vector<Ring> shares_to_send(2 * num_mult_gates);
        for (size_t i = 0; i < num_mult_gates; ++i) {
            shares_to_send[2*i] = u_shares[i];
            shares_to_send[2*i + 1] = v_shares[i];
        }
        
        std::vector<Ring> reconstructed(2 * num_mult_gates, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
        
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
            auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[mult_gate.out].get());
            
            Ring u = u_reconstructed[i];
            Ring v = v_reconstructed[i];
            Ring a = pre_out->triple_a.valueAt();
            Ring b = pre_out->triple_b.valueAt();
            Ring c = pre_out->triple_c.valueAt();
            
            // Beaver triple formula: z = u*v + u*b + v*a + c
            if (id_ == 1){
                wires_[mult_gate.out] = u * v + u * b + v * a + c;
            }
            else{
                wires_[mult_gate.out] = u * b + v * a + c;
            }
        }
    }


    void OnlineEvaluator::rewireEvaluate(const std::vector<common::utils::SIMDOGate> &rewire_gates) {
        if (id_ == 0) { return; }
        if (rewire_gates.empty()) { return; }

        size_t num_gates = rewire_gates.size();
        
        // Extract metadata and prepare data structures for all gates in parallel
        std::vector<size_t> vec_sizes(num_gates);
        std::vector<size_t> num_payloads_vec(num_gates);
        std::vector<std::vector<Ring>> all_position_maps(num_gates);
        std::vector<std::vector<std::vector<Ring>>> all_payload_shares(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            const auto& rewire_gate = rewire_gates[g];
            
            // Input format: [pos_map_0, ..., pos_map_n, p1_0, ..., p1_n, p2_0, ..., p2_n, ...]
            // Output format: [p1_out_0, ..., p1_out_n, p2_out_0, ..., p2_out_n, ...]
            // Note: position_map wires already hold reconstructed values (public)
            
            // Get vec_size and num_payloads from preprocessing
            auto* preproc_rewire = static_cast<PreprocRewireGate<Ring>*>(preproc_.gates[rewire_gate.out].get());
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
            all_payload_shares[g].resize(num_payloads, std::vector<Ring>(vec_size));
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    all_payload_shares[g][p][i] = wires_[rewire_gate.in[vec_size * (p + 1) + i]];
                }
            }
        }
        
        // Apply permutations for all gates in parallel
        std::vector<std::vector<std::unordered_map<common::utils::wire_t, Ring>>> all_temp_outputs(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            const auto& rewire_gate = rewire_gates[g];
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];
            
            all_temp_outputs[g].resize(num_payloads);
            
            // Apply public permutation based on position_map
            // For each position i: if position_map[i] = idx_perm, then output[idx_perm] = payload[i]
            for (size_t i = 0; i < vec_size; ++i) {
                int idx_perm = static_cast<int>(all_position_maps[g][i]);
                if (idx_perm >= 0 && idx_perm < vec_size) {
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

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }

        std::vector<common::utils::FIn2Gate> mult_gates;
        std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::FIn1Gate> rec_gates;
        std::vector<common::utils::SIMDOGate> shuffle_gates;
        std::vector<common::utils::SIMDOGate> permAndSh_gates;
        std::vector<common::utils::SIMDOGate> amortzdPnS_gates;
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
                case common::utils::GateType::kShuffle: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    shuffle_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kPermAndSh: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    permAndSh_gates.push_back(*g);
                    break;
                }

                case common::utils::GateType::kAmortzdPnS: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    amortzdPnS_gates.push_back(*g);
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
        if (!shuffle_gates.empty()) { shuffleEvaluate(shuffle_gates); }
        if (!permAndSh_gates.empty()) { permAndShEvaluate(permAndSh_gates); }
        if (!amortzdPnS_gates.empty()) { amortzdPnSEvaluate(amortzdPnS_gates); }
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
                    auto *g = static_cast<common::utils::ConstOpGate<Ring> *>(gate.get());
                    if (id_ == 1) {
                        wires_[g->out] = wires_[g->in] + g->cval;
                    } else {
                        wires_[g->out] = wires_[g->in];
                    }
                    break;
                }
                case common::utils::GateType::kConstMul: {
                    auto *g = static_cast<common::utils::ConstOpGate<Ring> *>(gate.get());
                    wires_[g->out] = wires_[g->in] * g->cval;
                    break;
                }
                case common::utils::GateType::kPublicPerm: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    auto vec_len = g->in.size();
                    
                    for (int i = 0; i < vec_len; ++i) {
                        std::cout << g->outs[i] << " ";
                    }
                    std::cout << std::endl;
                    
                    // First, read all inputs and compute permutation indices
                    std::unordered_map<common::utils::wire_t, Ring> temp_outputs;
                    for (int i = 0; i < vec_len; ++i) {
                        auto idx_perm = g->permutation[0][i];
                        if (idx_perm < 0) {
                            common::utils::wire_t wire_id = static_cast<common::utils::wire_t>(-idx_perm - 1);
                            Ring wire_value = wires_[wire_id];
                            idx_perm = static_cast<int>(wire_value);
                        }
                        temp_outputs[g->outs[idx_perm]] = wires_[g->in[i]];
                    }
                    
                    // Then, write all outputs at once
                    for (const auto& [wire_id, value] : temp_outputs) {
                        wires_[wire_id] = value;
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    std::vector<Ring> OnlineEvaluator::getOutputs() {
        std::vector<Ring> outvals(circ_.outputs.size());
        if (circ_.outputs.empty()) {
            return outvals;
        }
        if (id_ != 0) {
            std::vector<std::vector<Ring>> output_shares(nP_, std::vector<Ring>(circ_.outputs.size()));
            for (size_t i = 0; i < circ_.outputs.size(); ++i) {
                auto wout = circ_.outputs[i];
                output_shares[id_ - 1][i] = wires_[wout];
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->send(pid, output_shares[id_ - 1].data(), output_shares[id_ - 1].size() * sizeof(Ring));
                }
            }
            usleep(latency_);
            // #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->recv(pid, output_shares[pid - 1].data(), output_shares[pid - 1].size() * sizeof(Ring));
                }
            }
            for (size_t i = 0; i < circ_.outputs.size(); ++i) {
                Ring outmask = Ring(0);
                for (int pid = 1; pid <= nP_; ++pid) {
                    outmask += output_shares[pid - 1][i];
                }
                outvals[i] = outmask;
            }
        }
        return outvals;
    }

    std::vector<Ring> OnlineEvaluator::evaluateCircuit(const std::unordered_map<common::utils::wire_t, Ring> &inputs) {
        setInputs(inputs);
        for (size_t i = 0; i < circ_.gates_by_level.size(); ++i) {
            evaluateGatesAtDepth(i);
        }
        return getOutputs();
    }

}; // namespace graphdb
