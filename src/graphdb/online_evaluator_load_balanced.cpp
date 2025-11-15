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

    void OnlineEvaluator::compactEvaluate(const common::utils::SIMDOGate &compact_gate) {
        if (id_ == 0) { return; }
        
        auto *pre_compact = static_cast<PreprocCompactGate<Ring> *>(preproc_.gates[compact_gate.out].get());
        size_t total_size = compact_gate.in.size();
        // Determine vec_size and num_payloads from total_size
        // total_size = vec_size * (1 + num_payloads)
        // We can get this from the output size as well
        size_t total_outputs = compact_gate.outs.size();
        // Need to determine vec_size - it's stored in preprocessing or we can deduce it
        // From preprocessing, we know shuffle_a.size() gives us vec_size
        size_t vec_size = pre_compact->shuffle_a.size();
        size_t num_payloads = (total_size / vec_size) - 1;
        
        // Extract t and p vectors from inputs
        // Input format: [t0,...,tn, p1_0,...,p1_n, p2_0,...,p2_n, ...]
        std::vector<Ring> t_shares(vec_size);
        std::vector<std::vector<Ring>> p_shares(num_payloads, std::vector<Ring>(vec_size));
        
        for (size_t i = 0; i < vec_size; ++i) {
            t_shares[i] = wires_[compact_gate.in[i]];
        }
        for (size_t p = 0; p < num_payloads; ++p) {
            for (size_t i = 0; i < vec_size; ++i) {
                p_shares[p][i] = wires_[compact_gate.in[vec_size * (p + 1) + i]];
            }
        }
        
        // Step 1: Compute prefix sums c0 and c1 locally on shares
        std::vector<Ring> c1_shares(vec_size);
        std::vector<Ring> c0_shares(vec_size);
        
        c1_shares[0] = t_shares[0];
        if (id_ == 1) {
            c0_shares[0] = Ring(1) - c1_shares[0];
        } else {
            c0_shares[0] = -c1_shares[0];
        }
        
        for (size_t j = 1; j < vec_size; ++j) {
            c1_shares[j] = c1_shares[j-1] + t_shares[j];
            if (id_ == 1) {
                c0_shares[j] = Ring(j+1) - c1_shares[j];
            } else {
                c0_shares[j] = -c1_shares[j];
            }
        }
        
        // Step 2: Compute label shares
        // label[j] = (c0[j] + c1[N-1] - c1[j])(1 - t[j]) + c1[j] - 1
        std::vector<Ring> label_shares(vec_size);
        Ring c1_last = c1_shares[vec_size - 1];
        
        // We need to compute multiplications using Beaver triples
        // First, prepare masked values for reconstruction
        int pKing = 1;
        std::vector<Ring> u_shares(vec_size);  // diff_term - a
        std::vector<Ring> v_shares(vec_size);  // (1-t) - b
        
        for (size_t j = 0; j < vec_size; ++j) {
            // diff_term = c0[j] + c1_last - c1[j]
            Ring diff_term = c0_shares[j] + c1_last - c1_shares[j];
            
            // one_minus_t = 1 - t[j]
            Ring one_minus_t;
            if (id_ == 1) {
                one_minus_t = Ring(1) - t_shares[j];
            } else {
                one_minus_t = -t_shares[j];
            }
            
            // Masked values for multiplication
            u_shares[j] = diff_term - pre_compact->mult_triple_a[j].valueAt();
            v_shares[j] = one_minus_t - pre_compact->mult_triple_b[j].valueAt();
        }
        
        // Reconstruct u and v
        std::vector<Ring> u_reconstructed(vec_size, 0);
        std::vector<Ring> v_reconstructed(vec_size, 0);
        
        std::vector<Ring> shares_to_send(2 * vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            shares_to_send[2*i] = u_shares[i];
            shares_to_send[2*i + 1] = v_shares[i];
        }
        
        std::vector<Ring> reconstructed(2 * vec_size, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
        
        for (size_t i = 0; i < vec_size; ++i) {
            u_reconstructed[i] = reconstructed[2*i];
            v_reconstructed[i] = reconstructed[2*i + 1];
        }
        
        // Compute label shares
        for (size_t j = 0; j < vec_size; ++j) {
            Ring u = u_reconstructed[j];
            Ring v = v_reconstructed[j];
            Ring a = pre_compact->mult_triple_a[j].valueAt();
            Ring b = pre_compact->mult_triple_b[j].valueAt();
            Ring c = pre_compact->mult_triple_c[j].valueAt();
            
            Ring mult_result;
            if (id_ == 1) {
                mult_result = u * v + u * b + v * a + c;
                label_shares[j] = mult_result + c1_shares[j] - Ring(1);
            } else {
                mult_result = u * b + v * a + c;
                label_shares[j] = mult_result + c1_shares[j];
            }
        }
        
        // Step 3: Shuffle t, all p vectors, and label using the shuffle preprocessing
        std::vector<Ring> t_shuffled(vec_size);
        std::vector<std::vector<Ring>> p_shuffled(num_payloads, std::vector<Ring>(vec_size));
        std::vector<Ring> label_shuffled(vec_size);
        
        // Perform shuffle for all vectors (t, p1, p2, ..., label)
        // We use the same shuffle preprocessing for all
        
        // Compute z = input + a for each vector
        std::vector<Ring> z_t(vec_size);
        std::vector<std::vector<Ring>> z_p(num_payloads, std::vector<Ring>(vec_size));
        std::vector<Ring> z_label(vec_size);
        
        for (size_t i = 0; i < vec_size; ++i) {
            z_t[i] = t_shares[i] + pre_compact->shuffle_a[i].valueAt();
            z_label[i] = label_shares[i] + pre_compact->shuffle_a[i].valueAt();
        }
        for (size_t p = 0; p < num_payloads; ++p) {
            for (size_t i = 0; i < vec_size; ++i) {
                z_p[p][i] = p_shares[p][i] + pre_compact->shuffle_a[i].valueAt();
            }
        }
        
        if (id_ == 1) {
            // Party 1 collects z values from all parties
            std::vector<std::vector<Ring>> z_t_recv(nP_);
            std::vector<std::vector<std::vector<Ring>>> z_p_recv(nP_, std::vector<std::vector<Ring>>(num_payloads));
            std::vector<std::vector<Ring>> z_label_recv(nP_);
            
            z_t_recv[0] = z_t;
            z_p_recv[0] = z_p;
            z_label_recv[0] = z_label;
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_t_recv[pid - 1].resize(vec_size);
                z_label_recv[pid - 1].resize(vec_size);
                network_->recv(pid, z_t_recv[pid - 1].data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    z_p_recv[pid - 1][p].resize(vec_size);
                    network_->recv(pid, z_p_recv[pid - 1][p].data(), vec_size * sizeof(Ring));
                }
                network_->recv(pid, z_label_recv[pid - 1].data(), vec_size * sizeof(Ring));
            }
            usleep(latency_);
            
            // Sum all z values
            std::vector<Ring> z_t_sum(vec_size, 0);
            std::vector<std::vector<Ring>> z_p_sum(num_payloads, std::vector<Ring>(vec_size, 0));
            std::vector<Ring> z_label_sum(vec_size, 0);
            
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < vec_size; ++i) {
                    z_t_sum[i] += z_t_recv[pid][i];
                    z_label_sum[i] += z_label_recv[pid][i];
                }
                for (size_t p = 0; p < num_payloads; ++p) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_p_sum[p][i] += z_p_recv[pid][p][i];
                    }
                }
            }
            
            // Apply permutation and add b
            std::vector<Ring> z_t_perm(vec_size);
            std::vector<std::vector<Ring>> z_p_perm(num_payloads, std::vector<Ring>(vec_size));
            std::vector<Ring> z_label_perm(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_compact->shuffle_pi[i];
                z_t_perm[i] = z_t_sum[pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                z_label_perm[i] = z_label_sum[pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                
                // Set output shares to c
                t_shuffled[i] = pre_compact->shuffle_c[i].valueAt();
                label_shuffled[i] = pre_compact->shuffle_c[i].valueAt();
            }
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_compact->shuffle_pi[i];
                    z_p_perm[p][i] = z_p_sum[p][pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                    p_shuffled[p][i] = pre_compact->shuffle_c[i].valueAt();
                }
            }
            
            // Send to party 2
            network_->send(2, z_t_perm.data(), vec_size * sizeof(Ring));
            for (size_t p = 0; p < num_payloads; ++p) {
                network_->send(2, z_p_perm[p].data(), vec_size * sizeof(Ring));
            }
            network_->send(2, z_label_perm.data(), vec_size * sizeof(Ring));
            
        } else {
            // Parties 2 to nP
            // Send z values to party 1
            network_->send(1, z_t.data(), vec_size * sizeof(Ring));
            for (size_t p = 0; p < num_payloads; ++p) {
                network_->send(1, z_p[p].data(), vec_size * sizeof(Ring));
            }
            network_->send(1, z_label.data(), vec_size * sizeof(Ring));
            network_->flush(1);
            
            // Receive from previous party
            std::vector<Ring> z_t_recv(vec_size);
            std::vector<std::vector<Ring>> z_p_recv(num_payloads, std::vector<Ring>(vec_size));
            std::vector<Ring> z_label_recv(vec_size);
            
            network_->recv(id_ - 1, z_t_recv.data(), vec_size * sizeof(Ring));
            for (size_t p = 0; p < num_payloads; ++p) {
                network_->recv(id_ - 1, z_p_recv[p].data(), vec_size * sizeof(Ring));
            }
            network_->recv(id_ - 1, z_label_recv.data(), vec_size * sizeof(Ring));
            usleep(latency_);
            
            // Apply permutation and add b
            std::vector<Ring> z_t_send(vec_size);
            std::vector<std::vector<Ring>> z_p_send(num_payloads, std::vector<Ring>(vec_size));
            std::vector<Ring> z_label_send(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_compact->shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_t_send[i] = z_t_recv[pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                    z_label_send[i] = z_label_recv[pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                    
                    t_shuffled[i] = pre_compact->shuffle_c[i].valueAt();
                    label_shuffled[i] = pre_compact->shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_t_send[i] = z_t_recv[pi_i] + pre_compact->shuffle_b[pi_i].valueAt() - pre_compact->shuffle_delta[i];
                    z_label_send[i] = z_label_recv[pi_i] + pre_compact->shuffle_b[pi_i].valueAt() - pre_compact->shuffle_delta[i];
                    
                    t_shuffled[i] = z_t_send[i] + pre_compact->shuffle_c[i].valueAt();
                    label_shuffled[i] = z_label_send[i] + pre_compact->shuffle_c[i].valueAt();
                }
            }
            
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_compact->shuffle_pi[i];
                    
                    if (id_ != nP_) {
                        z_p_send[p][i] = z_p_recv[p][pi_i] + pre_compact->shuffle_b[pi_i].valueAt();
                        p_shuffled[p][i] = pre_compact->shuffle_c[i].valueAt();
                    } else {
                        z_p_send[p][i] = z_p_recv[p][pi_i] + pre_compact->shuffle_b[pi_i].valueAt() - pre_compact->shuffle_delta[i];
                        p_shuffled[p][i] = z_p_send[p][i] + pre_compact->shuffle_c[i].valueAt();
                    }
                }
            }
            
            // Send to next party (if not last)
            if (id_ != nP_) {
                network_->send(id_ + 1, z_t_send.data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->send(id_ + 1, z_p_send[p].data(), vec_size * sizeof(Ring));
                }
                network_->send(id_ + 1, z_label_send.data(), vec_size * sizeof(Ring));
            }
        }
        
        // Step 4: Reconstruct label_shuffled
        std::vector<Ring> label_reconstructed(vec_size, 0);
        reconstruct(nP_, id_, network_, label_shuffled, label_reconstructed, use_pking_, latency_);
        
        // Step 5: Apply public permutation based on reconstructed labels
        std::unordered_map<common::utils::wire_t, Ring> temp_t_outputs;
        std::vector<std::unordered_map<common::utils::wire_t, Ring>> temp_p_outputs(num_payloads);
        
        for (size_t i = 0; i < vec_size; ++i) {
            int idx_perm = static_cast<int>(label_reconstructed[i]);
            if (idx_perm >= 0 && idx_perm < vec_size) {
                temp_t_outputs[compact_gate.outs[idx_perm]] = t_shuffled[i];
                for (size_t p = 0; p < num_payloads; ++p) {
                    temp_p_outputs[p][compact_gate.outs[vec_size * (p + 1) + idx_perm]] = p_shuffled[p][i];
                }
            }
        }
        
        // Write outputs
        for (const auto& [wire_id, value] : temp_t_outputs) {
            wires_[wire_id] = value;
        }
        for (size_t p = 0; p < num_payloads; ++p) {
            for (const auto& [wire_id, value] : temp_p_outputs[p]) {
                wires_[wire_id] = value;
            }
        }
    }

    void OnlineEvaluator::compactEvaluateParallel(const std::vector<common::utils::SIMDOGate> &compact_gates) {
        if (id_ == 0) { return; }
        if (compact_gates.empty()) { return; }
        
        size_t num_gates = compact_gates.size();
        
        // Extract metadata for each gate
        std::vector<size_t> vec_sizes(num_gates);
        std::vector<size_t> num_payloads_vec(num_gates);
        std::vector<PreprocCompactGate<Ring>*> pre_compacts(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            pre_compacts[g] = static_cast<PreprocCompactGate<Ring>*>(preproc_.gates[compact_gates[g].out].get());
            vec_sizes[g] = pre_compacts[g]->shuffle_a.size();
            size_t total_size = compact_gates[g].in.size();
            num_payloads_vec[g] = (total_size / vec_sizes[g]) - 1;
        }
        
        // Extract input shares for all gates
        std::vector<std::vector<Ring>> all_t_shares(num_gates);
        std::vector<std::vector<std::vector<Ring>>> all_p_shares(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];
            
            all_t_shares[g].resize(vec_size);
            all_p_shares[g].resize(num_payloads, std::vector<Ring>(vec_size));
            
            for (size_t i = 0; i < vec_size; ++i) {
                all_t_shares[g][i] = wires_[compact_gates[g].in[i]];
            }
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    all_p_shares[g][p][i] = wires_[compact_gates[g].in[vec_size * (p + 1) + i]];
                }
            }
        }
        
        // Step 1: Compute prefix sums for all gates in parallel
        std::vector<std::vector<Ring>> all_c1_shares(num_gates);
        std::vector<std::vector<Ring>> all_c0_shares(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_c1_shares[g].resize(vec_size);
            all_c0_shares[g].resize(vec_size);
            
            all_c1_shares[g][0] = all_t_shares[g][0];
            if (id_ == 1) {
                all_c0_shares[g][0] = Ring(1) - all_c1_shares[g][0];
            } else {
                all_c0_shares[g][0] = -all_c1_shares[g][0];
            }
            
            for (size_t j = 1; j < vec_size; ++j) {
                all_c1_shares[g][j] = all_c1_shares[g][j-1] + all_t_shares[g][j];
                if (id_ == 1) {
                    all_c0_shares[g][j] = Ring(j+1) - all_c1_shares[g][j];
                } else {
                    all_c0_shares[g][j] = -all_c1_shares[g][j];
                }
            }
        }
        
        // Step 2: Prepare masked values for all gates
        size_t total_mults = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            total_mults += vec_sizes[g];
        }
        
        std::vector<Ring> all_u_shares;
        std::vector<Ring> all_v_shares;
        all_u_shares.reserve(total_mults);
        all_v_shares.reserve(total_mults);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            Ring c1_last = all_c1_shares[g][vec_size - 1];
            
            for (size_t j = 0; j < vec_size; ++j) {
                Ring diff_term = all_c0_shares[g][j] + c1_last - all_c1_shares[g][j];
                Ring one_minus_t;
                if (id_ == 1) {
                    one_minus_t = Ring(1) - all_t_shares[g][j];
                } else {
                    one_minus_t = -all_t_shares[g][j];
                }
                
                all_u_shares.push_back(diff_term - pre_compacts[g]->mult_triple_a[j].valueAt());
                all_v_shares.push_back(one_minus_t - pre_compacts[g]->mult_triple_b[j].valueAt());
            }
        }
        
        // Reconstruct u and v for all gates
        std::vector<Ring> shares_to_send(2 * total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            shares_to_send[2*i] = all_u_shares[i];
            shares_to_send[2*i + 1] = all_v_shares[i];
        }
        
        std::vector<Ring> reconstructed(2 * total_mults, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
        
        std::vector<Ring> u_reconstructed(total_mults);
        std::vector<Ring> v_reconstructed(total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            u_reconstructed[i] = reconstructed[2*i];
            v_reconstructed[i] = reconstructed[2*i + 1];
        }
        
        // Compute label shares for all gates
        std::vector<std::vector<Ring>> all_label_shares(num_gates);
        
        size_t mult_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_label_shares[g].resize(vec_size);
            
            for (size_t j = 0; j < vec_size; ++j) {
                Ring u = u_reconstructed[mult_offset + j];
                Ring v = v_reconstructed[mult_offset + j];
                Ring a = pre_compacts[g]->mult_triple_a[j].valueAt();
                Ring b = pre_compacts[g]->mult_triple_b[j].valueAt();
                Ring c = pre_compacts[g]->mult_triple_c[j].valueAt();
                
                Ring mult_result;
                if (id_ == 1) {
                    mult_result = u * v + u * b + v * a + c;
                    all_label_shares[g][j] = mult_result + all_c1_shares[g][j] - Ring(1);
                } else {
                    mult_result = u * b + v * a + c;
                    all_label_shares[g][j] = mult_result + all_c1_shares[g][j];
                }
            }
            mult_offset += vec_size;
        }
        
        // Step 3: Shuffle all vectors for all gates
        std::vector<std::vector<Ring>> all_t_shuffled(num_gates);
        std::vector<std::vector<std::vector<Ring>>> all_p_shuffled(num_gates);
        std::vector<std::vector<Ring>> all_label_shuffled(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];
            
            all_t_shuffled[g].resize(vec_size);
            all_p_shuffled[g].resize(num_payloads, std::vector<Ring>(vec_size));
            all_label_shuffled[g].resize(vec_size);
        }
        
        // Compute z = input + a for all gates
        std::vector<std::vector<Ring>> all_z_t(num_gates);
        std::vector<std::vector<std::vector<Ring>>> all_z_p(num_gates);
        std::vector<std::vector<Ring>> all_z_label(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];
            
            all_z_t[g].resize(vec_size);
            all_z_p[g].resize(num_payloads, std::vector<Ring>(vec_size));
            all_z_label[g].resize(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                all_z_t[g][i] = all_t_shares[g][i] + pre_compacts[g]->shuffle_a[i].valueAt();
                all_z_label[g][i] = all_label_shares[g][i] + pre_compacts[g]->shuffle_a[i].valueAt();
            }
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    all_z_p[g][p][i] = all_p_shares[g][p][i] + pre_compacts[g]->shuffle_a[i].valueAt();
                }
            }
        }
        
        if (id_ == 1) {
            // Party 1 collects and accumulates z values from all parties for each gate without storing full copies
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                size_t num_payloads = num_payloads_vec[g];

                std::vector<Ring> z_t_sum(vec_size, 0);
                std::vector<std::vector<Ring>> z_p_sum(num_payloads, std::vector<Ring>(vec_size, 0));
                std::vector<Ring> z_label_sum(vec_size, 0);

                // Start with local contributions
                for (size_t i = 0; i < vec_size; ++i) {
                    z_t_sum[i] = all_z_t[g][i];
                    z_label_sum[i] = all_z_label[g][i];
                }
                for (size_t p = 0; p < num_payloads; ++p) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_p_sum[p][i] = all_z_p[g][p][i];
                    }
                }

                // Temporary buffers reused for each incoming party
                std::vector<Ring> z_t_recv(vec_size);
                std::vector<std::vector<Ring>> z_p_recv(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> z_label_recv(vec_size);

                #pragma omp parallel for
                for (int pid = 2; pid <= nP_; ++pid) {
                    network_->recv(pid, z_t_recv.data(), vec_size * sizeof(Ring));
                    for (size_t p = 0; p < num_payloads; ++p) {
                        network_->recv(pid, z_p_recv[p].data(), vec_size * sizeof(Ring));
                    }
                    network_->recv(pid, z_label_recv.data(), vec_size * sizeof(Ring));

                    for (size_t i = 0; i < vec_size; ++i) {
                        z_t_sum[i] += z_t_recv[i];
                        z_label_sum[i] += z_label_recv[i];
                    }
                    for (size_t p = 0; p < num_payloads; ++p) {
                        for (size_t i = 0; i < vec_size; ++i) {
                            z_p_sum[p][i] += z_p_recv[p][i];
                        }
                    }
                }
                usleep(latency_);

                std::vector<Ring> z_t_perm(vec_size);
                std::vector<std::vector<Ring>> z_p_perm(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> z_label_perm(vec_size);

                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_compacts[g]->shuffle_pi[i];
                    z_t_perm[i] = z_t_sum[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();
                    z_label_perm[i] = z_label_sum[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();

                    all_t_shuffled[g][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                    all_label_shuffled[g][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                }
                for (size_t p = 0; p < num_payloads; ++p) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        int pi_i = pre_compacts[g]->shuffle_pi[i];
                        z_p_perm[p][i] = z_p_sum[p][pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();
                        all_p_shuffled[g][p][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                    }
                }

                // Send to party 2
                network_->send(2, z_t_perm.data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->send(2, z_p_perm[p].data(), vec_size * sizeof(Ring));
                }
                network_->send(2, z_label_perm.data(), vec_size * sizeof(Ring));
            }
            network_->flush(2);

        } else {
            // Parties 2 to nP
            // Send z values to party 1 for all gates
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                size_t num_payloads = num_payloads_vec[g];
                
                network_->send(1, all_z_t[g].data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->send(1, all_z_p[g][p].data(), vec_size * sizeof(Ring));
                }
                network_->send(1, all_z_label[g].data(), vec_size * sizeof(Ring));
            }
            network_->flush(1);
            
            // Process all gates
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                size_t num_payloads = num_payloads_vec[g];
                
                std::vector<Ring> z_t_recv(vec_size);
                std::vector<std::vector<Ring>> z_p_recv(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> z_label_recv(vec_size);
                
                network_->recv(id_ - 1, z_t_recv.data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->recv(id_ - 1, z_p_recv[p].data(), vec_size * sizeof(Ring));
                }
                network_->recv(id_ - 1, z_label_recv.data(), vec_size * sizeof(Ring));
                usleep(latency_);
                
                std::vector<Ring> z_t_send(vec_size);
                std::vector<std::vector<Ring>> z_p_send(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> z_label_send(vec_size);
                
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_compacts[g]->shuffle_pi[i];
                    
                    if (id_ != nP_) {
                        z_t_send[i] = z_t_recv[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();
                        z_label_send[i] = z_label_recv[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();
                        
                        all_t_shuffled[g][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                        all_label_shuffled[g][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                    } else {
                        z_t_send[i] = z_t_recv[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt() - pre_compacts[g]->shuffle_delta[i];
                        z_label_send[i] = z_label_recv[pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt() - pre_compacts[g]->shuffle_delta[i];
                        
                        all_t_shuffled[g][i] = z_t_send[i] + pre_compacts[g]->shuffle_c[i].valueAt();
                        all_label_shuffled[g][i] = z_label_send[i] + pre_compacts[g]->shuffle_c[i].valueAt();
                    }
                }
                
                for (size_t p = 0; p < num_payloads; ++p) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        int pi_i = pre_compacts[g]->shuffle_pi[i];
                        
                        if (id_ != nP_) {
                            z_p_send[p][i] = z_p_recv[p][pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt();
                            all_p_shuffled[g][p][i] = pre_compacts[g]->shuffle_c[i].valueAt();
                        } else {
                            z_p_send[p][i] = z_p_recv[p][pi_i] + pre_compacts[g]->shuffle_b[pi_i].valueAt() - pre_compacts[g]->shuffle_delta[i];
                            all_p_shuffled[g][p][i] = z_p_send[p][i] + pre_compacts[g]->shuffle_c[i].valueAt();
                        }
                    }
                }
                
                if (id_ != nP_) {
                    network_->send(id_ + 1, z_t_send.data(), vec_size * sizeof(Ring));
                    for (size_t p = 0; p < num_payloads; ++p) {
                        network_->send(id_ + 1, z_p_send[p].data(), vec_size * sizeof(Ring));
                    }
                    network_->send(id_ + 1, z_label_send.data(), vec_size * sizeof(Ring));
                }
            }
            if (id_ != nP_) {
                network_->flush(id_ + 1);
            }
        }
        
        // Step 4: Reconstruct labels for all gates in batch
        size_t total_labels = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            total_labels += vec_sizes[g];
        }
        
        std::vector<Ring> all_labels_flat;
        all_labels_flat.reserve(total_labels);
        for (size_t g = 0; g < num_gates; ++g) {
            all_labels_flat.insert(all_labels_flat.end(), 
                                  all_label_shuffled[g].begin(), 
                                  all_label_shuffled[g].end());
        }
        
        std::vector<Ring> all_labels_reconstructed(total_labels, 0);
        reconstruct(nP_, id_, network_, all_labels_flat, all_labels_reconstructed, use_pking_, latency_);
        
        // Step 5: Apply public permutation for all gates
        size_t label_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            size_t num_payloads = num_payloads_vec[g];

            for (size_t i = 0; i < vec_size; ++i) {
                Ring label = all_labels_reconstructed[label_offset + i];
                if (label < 0 || label >= static_cast<Ring>(vec_size)) {
                    continue;
                }
                size_t idx_perm = static_cast<size_t>(label);

                auto t_out_wire = compact_gates[g].outs[idx_perm];
                wires_[t_out_wire] = all_t_shuffled[g][i];

                for (size_t p = 0; p < num_payloads; ++p) {
                    auto payload_out_wire = compact_gates[g].outs[vec_size * (p + 1) + idx_perm];
                    wires_[payload_out_wire] = all_p_shuffled[g][p][i];
                }
            }

            label_offset += vec_size;
        }
    }

    void OnlineEvaluator::groupwiseIndexEvaluate(const common::utils::SIMDOGate &gi_gate) {
        if (id_ == 0) { return; }
        
        auto *pre_gi = static_cast<PreprocGroupwiseIndexGate<Ring> *>(preproc_.gates[gi_gate.out].get());
        
        // Input format: [key0,...,keyn, v0,...,vn]
        // Output format: [ind0,...,indn, key0,...,keyn, v0,...,vn]
        size_t vec_size = gi_gate.in.size() / 2;
        
        // Extract key and v vectors from inputs
        std::vector<Ring> key_shares(vec_size);
        std::vector<Ring> v_shares(vec_size);
        
        for (size_t i = 0; i < vec_size; ++i) {
            key_shares[i] = wires_[gi_gate.in[i]];
            v_shares[i] = wires_[gi_gate.in[vec_size + i]];
        }
        
        // Step 1: Initialize ind vector locally (shares of 0, 1, 2, ..., N-1)
        // and perform first compaction with key as tag and ind as payload
        std::vector<Ring> ind_shares(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            if (id_ == 1) {
                ind_shares[i] = Ring(i);
            } else {
                ind_shares[i] = Ring(0);
            }
        }
        
        // Step 1a: Compute prefix sums c0 and c1 locally on key shares
        std::vector<Ring> c1_shares(vec_size);
        std::vector<Ring> c0_shares(vec_size);
        
        c1_shares[0] = key_shares[0];
        if (id_ == 1) {
            c0_shares[0] = Ring(1) - c1_shares[0];
        } else {
            c0_shares[0] = -c1_shares[0];
        }
        
        for (size_t j = 1; j < vec_size; ++j) {
            c1_shares[j] = c1_shares[j-1] + key_shares[j];
            if (id_ == 1) {
                c0_shares[j] = Ring(j+1) - c1_shares[j];
            } else {
                c0_shares[j] = -c1_shares[j];
            }
        }
        
        // Step 1b: Compute label shares using multiplication triples
        // label[j] = (c0[j] + c1[N-1] - c1[j])(1 - key[j]) + c1[j] - 1
        std::vector<Ring> label_shares(vec_size);
        Ring c1_last = c1_shares[vec_size - 1];
        
        int pKing = 1;
        std::vector<Ring> u_mult_shares(vec_size);  // diff_term - a
        std::vector<Ring> v_mult_shares(vec_size);  // (1-key) - b
        
        for (size_t j = 0; j < vec_size; ++j) {
            Ring diff_term = c0_shares[j] + c1_last - c1_shares[j];
            Ring one_minus_key;
            if (id_ == 1) {
                one_minus_key = Ring(1) - key_shares[j];
            } else {
                one_minus_key = -key_shares[j];
            }
            
            u_mult_shares[j] = diff_term - pre_gi->mult_triple_a[j].valueAt();
            v_mult_shares[j] = one_minus_key - pre_gi->mult_triple_b[j].valueAt();
        }
        
        // Step 1c: Reconstruct u and v via king party
        std::vector<Ring> u_reconstructed(vec_size, 0);
        std::vector<Ring> v_reconstructed(vec_size, 0);
        
        std::vector<Ring> shares_to_send(2 * vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            shares_to_send[2*i] = u_mult_shares[i];
            shares_to_send[2*i + 1] = v_mult_shares[i];
        }
        
        std::vector<Ring> reconstructed(2 * vec_size, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
        
        for (size_t i = 0; i < vec_size; ++i) {
            u_reconstructed[i] = reconstructed[2*i];
            v_reconstructed[i] = reconstructed[2*i + 1];
        }
        
        // Step 1d: Compute label shares using Beaver triple formula
        for (size_t j = 0; j < vec_size; ++j) {
            Ring u = u_reconstructed[j];
            Ring v = v_reconstructed[j];
            Ring a = pre_gi->mult_triple_a[j].valueAt();
            Ring b = pre_gi->mult_triple_b[j].valueAt();
            Ring c = pre_gi->mult_triple_c[j].valueAt();
            
            Ring mult_result;
            if (id_ == 1) {
                mult_result = u * v + u * b + v * a + c;
                label_shares[j] = mult_result + c1_shares[j] - Ring(1);
            } else {
                mult_result = u * b + v * a + c;
                label_shares[j] = mult_result + c1_shares[j];
            }
        }
        
        // Step 1e: Perform shuffle on key and ind using the shuffle preprocessing
        std::vector<Ring> key_shuffled(vec_size);
        std::vector<Ring> ind_shuffled(vec_size);
        std::vector<Ring> label_shuffled(vec_size);
        
        // Compute z = input + a for each vector
        std::vector<Ring> z_key(vec_size);
        std::vector<Ring> z_ind(vec_size);
        std::vector<Ring> z_label(vec_size);
        
        for (size_t i = 0; i < vec_size; ++i) {
            z_key[i] = key_shares[i] + pre_gi->shuffle_a[i].valueAt();
            z_ind[i] = ind_shares[i] + pre_gi->shuffle_a[i].valueAt();
            z_label[i] = label_shares[i] + pre_gi->shuffle_a[i].valueAt();
        }
        
        if (id_ == 1) {
            // Party 1 collects z values from all parties
            std::vector<std::vector<Ring>> z_key_recv(nP_);
            std::vector<std::vector<Ring>> z_ind_recv(nP_);
            std::vector<std::vector<Ring>> z_label_recv(nP_);
            
            z_key_recv[0] = z_key;
            z_ind_recv[0] = z_ind;
            z_label_recv[0] = z_label;
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_key_recv[pid - 1].resize(vec_size);
                z_ind_recv[pid - 1].resize(vec_size);
                z_label_recv[pid - 1].resize(vec_size);
                network_->recv(pid, z_key_recv[pid - 1].data(), vec_size * sizeof(Ring));
                network_->recv(pid, z_ind_recv[pid - 1].data(), vec_size * sizeof(Ring));
                network_->recv(pid, z_label_recv[pid - 1].data(), vec_size * sizeof(Ring));
            }
            usleep(latency_);
            
            // Sum all z values
            std::vector<Ring> z_key_sum(vec_size, 0);
            std::vector<Ring> z_ind_sum(vec_size, 0);
            std::vector<Ring> z_label_sum(vec_size, 0);
            
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < vec_size; ++i) {
                    z_key_sum[i] += z_key_recv[pid][i];
                    z_ind_sum[i] += z_ind_recv[pid][i];
                    z_label_sum[i] += z_label_recv[pid][i];
                }
            }
            
            // Apply permutation and add b
            std::vector<Ring> z_key_perm(vec_size);
            std::vector<Ring> z_ind_perm(vec_size);
            std::vector<Ring> z_label_perm(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_gi->shuffle_pi[i];
                z_key_perm[i] = z_key_sum[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                z_ind_perm[i] = z_ind_sum[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                z_label_perm[i] = z_label_sum[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                
                key_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
                ind_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
                label_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
            }
            
            // Send to party 2
            network_->send(2, z_key_perm.data(), vec_size * sizeof(Ring));
            network_->send(2, z_ind_perm.data(), vec_size * sizeof(Ring));
            network_->send(2, z_label_perm.data(), vec_size * sizeof(Ring));
            network_->flush(2);
            
        } else {
            // Parties 2 to nP: send z values to party 1
            network_->send(1, z_key.data(), vec_size * sizeof(Ring));
            network_->send(1, z_ind.data(), vec_size * sizeof(Ring));
            network_->send(1, z_label.data(), vec_size * sizeof(Ring));
            network_->flush(1);
            
            // Receive from previous party
            std::vector<Ring> z_key_recv(vec_size);
            std::vector<Ring> z_ind_recv(vec_size);
            std::vector<Ring> z_label_recv(vec_size);
            
            network_->recv(id_ - 1, z_key_recv.data(), vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_ind_recv.data(), vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_label_recv.data(), vec_size * sizeof(Ring));
            usleep(latency_);
            
            // Apply permutation and add b
            std::vector<Ring> z_key_send(vec_size);
            std::vector<Ring> z_ind_send(vec_size);
            std::vector<Ring> z_label_send(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_gi->shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_key_send[i] = z_key_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                    z_ind_send[i] = z_ind_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                    z_label_send[i] = z_label_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt();
                    
                    key_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
                    ind_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
                    label_shuffled[i] = pre_gi->shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_key_send[i] = z_key_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt() - pre_gi->shuffle_delta[i];
                    z_ind_send[i] = z_ind_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt() - pre_gi->shuffle_delta[i];
                    z_label_send[i] = z_label_recv[pi_i] + pre_gi->shuffle_b[pi_i].valueAt() - pre_gi->shuffle_delta[i];
                    
                    key_shuffled[i] = z_key_send[i] + pre_gi->shuffle_c[i].valueAt();
                    ind_shuffled[i] = z_ind_send[i] + pre_gi->shuffle_c[i].valueAt();
                    label_shuffled[i] = z_label_send[i] + pre_gi->shuffle_c[i].valueAt();
                }
            }
            
            // Send to next party (if not last)
            if (id_ != nP_) {
                network_->send(id_ + 1, z_key_send.data(), vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_ind_send.data(), vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_label_send.data(), vec_size * sizeof(Ring));
                network_->flush(id_ + 1);
            }
        }
        
        // Step 1f: Reconstruct label_shuffled to determine final placement
        std::vector<Ring> label_reconstructed(vec_size, 0);
        reconstruct(nP_, id_, network_, label_shuffled, label_reconstructed, use_pking_, latency_);
        
        // Step 1g: Apply public permutation based on reconstructed labels to get compacted vectors
        std::vector<Ring> key_compacted(vec_size, Ring(0));
        std::vector<Ring> ind_compacted(vec_size, Ring(0));
        
        for (size_t i = 0; i < vec_size; ++i) {
            Ring label = label_reconstructed[i];
            if (label >= 0 && label < static_cast<Ring>(vec_size)) {
                size_t idx_perm = static_cast<size_t>(label);
                key_compacted[idx_perm] = key_shuffled[i];
                ind_compacted[idx_perm] = ind_shuffled[i];
            }
        }
        
        // Step 2: Compute key_c locally (difference of consecutive ind_compacted values)
        // key_c[i] = ind_compacted[i-1] - ind_compacted[i]
        // key_c[0] = ind_compacted[0] (no predecessor)
        std::vector<Ring> key_c_shares(vec_size);
        key_c_shares[0] = ind_compacted[0];
        for (size_t i = 1; i < vec_size; ++i) {
            key_c_shares[i] = ind_compacted[i-1] - ind_compacted[i];
        }
        
        // Step 3: Compute ind_diff = key_c * key_compacted using multiplication triples
        // This tells us which positions mark group boundaries (ind_diff = ind where key_c != 0)
        std::vector<Ring> ind_diff_shares(vec_size);
        
        // Step 3a: Compute masked values for multiplication
        std::vector<Ring> u_keymult_shares(vec_size);  // key_c - a
        std::vector<Ring> v_keymult_shares(vec_size);  // key_compacted - b
        
        for (size_t j = 0; j < vec_size; ++j) {
            u_keymult_shares[j] = key_c_shares[j] - pre_gi->keymult_triple_a[j].valueAt();
            v_keymult_shares[j] = key_compacted[j] - pre_gi->keymult_triple_b[j].valueAt();
        }
        
        // Step 3b: Reconstruct u and v via king party
        std::vector<Ring> u_keymult_reconstructed(vec_size, 0);
        std::vector<Ring> v_keymult_reconstructed(vec_size, 0);
        
        std::vector<Ring> keymult_shares_to_send(2 * vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            keymult_shares_to_send[2*i] = u_keymult_shares[i];
            keymult_shares_to_send[2*i + 1] = v_keymult_shares[i];
        }
        
        std::vector<Ring> keymult_reconstructed(2 * vec_size, 0);
        reconstruct(nP_, id_, network_, keymult_shares_to_send, keymult_reconstructed, use_pking_, latency_);
        
        for (size_t i = 0; i < vec_size; ++i) {
            u_keymult_reconstructed[i] = keymult_reconstructed[2*i];
            v_keymult_reconstructed[i] = keymult_reconstructed[2*i + 1];
        }
        
        // Step 3c: Compute multiplication result using Beaver triple formula
        for (size_t j = 0; j < vec_size; ++j) {
            Ring u = u_keymult_reconstructed[j];
            Ring v = v_keymult_reconstructed[j];
            Ring a = pre_gi->keymult_triple_a[j].valueAt();
            Ring b = pre_gi->keymult_triple_b[j].valueAt();
            Ring c = pre_gi->keymult_triple_c[j].valueAt();
            
            if (id_ == 1) {
                ind_diff_shares[j] = u * v + u * b + v * a + c;
            } else {
                ind_diff_shares[j] = u * b + v * a + c;
            }
        }
        
        // Step 4: Apply inverse public permutation based on reconstructed labels
        // This reverses Step 1g by mapping compacted positions back to original positions
        std::vector<Ring> ind_diff_restored(vec_size, Ring(0));
        
        for (size_t i = 0; i < vec_size; ++i) {
            int label = static_cast<int>(label_reconstructed[i]);
            if (label >= 0 && label < static_cast<int>(vec_size)) {
                // The element at compacted position 'label' came from original position 'i'
                // So we restore it: ind_diff_restored[i] = ind_diff_shares[label]
                ind_diff_restored[i] = ind_diff_shares[label];
            }
        }
        
        // Step 5: Apply shuffle gate on ind_diff_restored to get final ind_diff
        // This is a second compaction step using reverse compaction preprocessing
        std::vector<Ring> ind_diff_final(vec_size);
        
        // Step 5a: Compute z = ind_diff_restored + a
        std::vector<Ring> z_final(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            z_final[i] = ind_diff_restored[i] + pre_gi->revcompact_shuffle_a[i].valueAt();
        }
        
        if (id_ == 1) {
            // Party 1: Collect z from all parties, apply permutation, and send to party 2
            std::vector<std::vector<Ring>> z_final_recv(nP_);
            z_final_recv[0] = z_final;
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_final_recv[pid - 1].resize(vec_size);
                network_->recv(pid, z_final_recv[pid - 1].data(), vec_size * sizeof(Ring));
            }
            usleep(latency_);
            
            // Sum all z values
            std::vector<Ring> z_final_sum(vec_size, 0);
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < vec_size; ++i) {
                    z_final_sum[i] += z_final_recv[pid][i];
                }
            }
            
            // Apply permutation and add b
            std::vector<Ring> z_final_perm(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_gi->revcompact_shuffle_pi[i];
                z_final_perm[i] = z_final_sum[pi_i] + pre_gi->revcompact_shuffle_b[pi_i].valueAt();
                ind_diff_final[i] = pre_gi->revcompact_shuffle_c[i].valueAt();
            }
            
            // Send to party 2
            network_->send(2, z_final_perm.data(), vec_size * sizeof(Ring));
            network_->flush(2);
            
        } else {
            // Parties 2 to nP: Send z to party 1
            network_->send(1, z_final.data(), vec_size * sizeof(Ring));
            network_->flush(1);
            
            // Receive from previous party
            std::vector<Ring> z_final_recv(vec_size);
            network_->recv(id_ - 1, z_final_recv.data(), vec_size * sizeof(Ring));
            usleep(latency_);
            
            // Apply permutation and add b
            std::vector<Ring> z_final_send(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                int pi_i = pre_gi->revcompact_shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_final_send[i] = z_final_recv[pi_i] + pre_gi->revcompact_shuffle_b[pi_i].valueAt();
                    ind_diff_final[i] = pre_gi->revcompact_shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_final_send[i] = z_final_recv[pi_i] + pre_gi->revcompact_shuffle_b[pi_i].valueAt() - pre_gi->revcompact_shuffle_delta[i];
                    ind_diff_final[i] = z_final_send[i] + pre_gi->revcompact_shuffle_c[i].valueAt();
                }
            }
            
            // Send to next party (if not last)
            if (id_ != nP_) {
                network_->send(id_ + 1, z_final_send.data(), vec_size * sizeof(Ring));
                network_->flush(id_ + 1);
            }
        }
        
        // Step 6: Compute prefix sum to get final group-wise indices
        // Each element's index within its group = cumulative sum of ind_diff up to that position
        std::vector<Ring> group_index_shares(vec_size);
        
        // Compute prefix sum locally on shares
        Ring running_sum = 0;
        for (size_t i = 0; i < vec_size; ++i) {
            running_sum += ind_diff_final[i];
            if (id_ == 1) {
                group_index_shares[i] = running_sum + Ring(i);
            } else {
                group_index_shares[i] = running_sum;
            }
        }
        
        // Write final outputs
        // Output format: [group_index0,...,group_indexn, key0,...,keyn, v0,...,vn]
        for (size_t i = 0; i < vec_size; ++i) {
            wires_[gi_gate.outs[i]] = group_index_shares[i];
            wires_[gi_gate.outs[vec_size + i]] = key_shares[i];
            wires_[gi_gate.outs[2 * vec_size + i]] = v_shares[i];
        }
        
    }

    void OnlineEvaluator::groupwiseIndexEvaluateParallel(const std::vector<common::utils::SIMDOGate> &gi_gates) {
        if (id_ == 0) { return; }
        if (gi_gates.empty()) { return; }
        
        size_t num_gates = gi_gates.size();
        
        // Extract metadata for each gate
        std::vector<size_t> vec_sizes(num_gates);
        std::vector<PreprocGroupwiseIndexGate<Ring>*> pre_gis(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            pre_gis[g] = static_cast<PreprocGroupwiseIndexGate<Ring>*>(preproc_.gates[gi_gates[g].out].get());
            vec_sizes[g] = gi_gates[g].in.size() / 2;
        }
        
        // Extract input shares for all gates
        std::vector<std::vector<Ring>> all_key_shares(num_gates);
        std::vector<std::vector<Ring>> all_v_shares(num_gates);
        std::vector<std::vector<Ring>> all_ind_shares(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_key_shares[g].resize(vec_size);
            all_v_shares[g].resize(vec_size);
            all_ind_shares[g].resize(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                all_key_shares[g][i] = wires_[gi_gates[g].in[i]];
                all_v_shares[g][i] = wires_[gi_gates[g].in[vec_size + i]];
                if (id_ == 1) {
                    all_ind_shares[g][i] = Ring(i);
                } else {
                    all_ind_shares[g][i] = Ring(0);
                }
            }
        }
        
        // Step 1: Compute prefix sums for all gates in parallel
        std::vector<std::vector<Ring>> all_c1_shares(num_gates);
        std::vector<std::vector<Ring>> all_c0_shares(num_gates);
        
        #pragma omp parallel for
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_c1_shares[g].resize(vec_size);
            all_c0_shares[g].resize(vec_size);
            
            all_c1_shares[g][0] = all_key_shares[g][0];
            if (id_ == 1) {
                all_c0_shares[g][0] = Ring(1) - all_c1_shares[g][0];
            } else {
                all_c0_shares[g][0] = -all_c1_shares[g][0];
            }
            
            for (size_t j = 1; j < vec_size; ++j) {
                all_c1_shares[g][j] = all_c1_shares[g][j-1] + all_key_shares[g][j];
                if (id_ == 1) {
                    all_c0_shares[g][j] = Ring(j+1) - all_c1_shares[g][j];
                } else {
                    all_c0_shares[g][j] = -all_c1_shares[g][j];
                }
            }
        }
        
        // Step 2: Prepare masked values for first multiplication (label computation)
        size_t total_mults = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            total_mults += vec_sizes[g];
        }
        
        std::vector<Ring> all_u_mult_shares;
        std::vector<Ring> all_v_mult_shares;
        all_u_mult_shares.reserve(total_mults);
        all_v_mult_shares.reserve(total_mults);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            Ring c1_last = all_c1_shares[g][vec_size - 1];
            
            for (size_t j = 0; j < vec_size; ++j) {
                Ring diff_term = all_c0_shares[g][j] + c1_last - all_c1_shares[g][j];
                Ring one_minus_key;
                if (id_ == 1) {
                    one_minus_key = Ring(1) - all_key_shares[g][j];
                } else {
                    one_minus_key = -all_key_shares[g][j];
                }
                
                all_u_mult_shares.push_back(diff_term - pre_gis[g]->mult_triple_a[j].valueAt());
                all_v_mult_shares.push_back(one_minus_key - pre_gis[g]->mult_triple_b[j].valueAt());
            }
        }
        
        // Reconstruct u and v for all gates
        std::vector<Ring> shares_to_send(2 * total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            shares_to_send[2*i] = all_u_mult_shares[i];
            shares_to_send[2*i + 1] = all_v_mult_shares[i];
        }
        
        std::vector<Ring> reconstructed(2 * total_mults, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
        
        std::vector<Ring> u_reconstructed(total_mults);
        std::vector<Ring> v_reconstructed(total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            u_reconstructed[i] = reconstructed[2*i];
            v_reconstructed[i] = reconstructed[2*i + 1];
        }
        
        // Compute label shares for all gates
        std::vector<std::vector<Ring>> all_label_shares(num_gates);
        
        size_t mult_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_label_shares[g].resize(vec_size);
            
            for (size_t j = 0; j < vec_size; ++j) {
                Ring u = u_reconstructed[mult_offset + j];
                Ring v = v_reconstructed[mult_offset + j];
                Ring a = pre_gis[g]->mult_triple_a[j].valueAt();
                Ring b = pre_gis[g]->mult_triple_b[j].valueAt();
                Ring c = pre_gis[g]->mult_triple_c[j].valueAt();
                
                Ring mult_result;
                if (id_ == 1) {
                    mult_result = u * v + u * b + v * a + c;
                    all_label_shares[g][j] = mult_result + all_c1_shares[g][j] - Ring(1);
                } else {
                    mult_result = u * b + v * a + c;
                    all_label_shares[g][j] = mult_result + all_c1_shares[g][j];
                }
            }
            mult_offset += vec_size;
        }
        
        // Step 3: Shuffle for all gates (first compaction)
        std::vector<std::vector<Ring>> all_key_shuffled(num_gates);
        std::vector<std::vector<Ring>> all_ind_shuffled(num_gates);
        std::vector<std::vector<Ring>> all_label_shuffled(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_key_shuffled[g].resize(vec_size);
            all_ind_shuffled[g].resize(vec_size);
            all_label_shuffled[g].resize(vec_size);
        }
        
        // Compute z values for all gates
        std::vector<std::vector<Ring>> all_z_key(num_gates);
        std::vector<std::vector<Ring>> all_z_ind(num_gates);
        std::vector<std::vector<Ring>> all_z_label(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_z_key[g].resize(vec_size);
            all_z_ind[g].resize(vec_size);
            all_z_label[g].resize(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                all_z_key[g][i] = all_key_shares[g][i] + pre_gis[g]->shuffle_a[i].valueAt();
                all_z_ind[g][i] = all_ind_shares[g][i] + pre_gis[g]->shuffle_a[i].valueAt();
                all_z_label[g][i] = all_label_shares[g][i] + pre_gis[g]->shuffle_a[i].valueAt();
            }
        }
        
        if (id_ == 1) {
            // Party 1 collects z values from all parties for all gates
            std::vector<std::vector<std::vector<Ring>>> all_z_key_recv(num_gates, std::vector<std::vector<Ring>>(nP_));
            std::vector<std::vector<std::vector<Ring>>> all_z_ind_recv(num_gates, std::vector<std::vector<Ring>>(nP_));
            std::vector<std::vector<std::vector<Ring>>> all_z_label_recv(num_gates, std::vector<std::vector<Ring>>(nP_));
            
            for (size_t g = 0; g < num_gates; ++g) {
                all_z_key_recv[g][0] = all_z_key[g];
                all_z_ind_recv[g][0] = all_z_ind[g];
                all_z_label_recv[g][0] = all_z_label[g];
            }
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                for (size_t g = 0; g < num_gates; ++g) {
                    size_t vec_size = vec_sizes[g];
                    all_z_key_recv[g][pid - 1].resize(vec_size);
                    all_z_ind_recv[g][pid - 1].resize(vec_size);
                    all_z_label_recv[g][pid - 1].resize(vec_size);
                    network_->recv(pid, all_z_key_recv[g][pid - 1].data(), vec_size * sizeof(Ring));
                    network_->recv(pid, all_z_ind_recv[g][pid - 1].data(), vec_size * sizeof(Ring));
                    network_->recv(pid, all_z_label_recv[g][pid - 1].data(), vec_size * sizeof(Ring));
                }
            }
            usleep(latency_);
            
            // Sum and permute for all gates
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                
                std::vector<Ring> z_key_sum(vec_size, 0);
                std::vector<Ring> z_ind_sum(vec_size, 0);
                std::vector<Ring> z_label_sum(vec_size, 0);
                
                for (int pid = 0; pid < nP_; ++pid) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_key_sum[i] += all_z_key_recv[g][pid][i];
                        z_ind_sum[i] += all_z_ind_recv[g][pid][i];
                        z_label_sum[i] += all_z_label_recv[g][pid][i];
                    }
                }
                
                std::vector<Ring> z_key_perm(vec_size);
                std::vector<Ring> z_ind_perm(vec_size);
                std::vector<Ring> z_label_perm(vec_size);
                
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_gis[g]->shuffle_pi[i];
                    z_key_perm[i] = z_key_sum[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                    z_ind_perm[i] = z_ind_sum[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                    z_label_perm[i] = z_label_sum[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                    
                    all_key_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                    all_ind_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                    all_label_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                }
                
                // Send to party 2
                network_->send(2, z_key_perm.data(), vec_size * sizeof(Ring));
                network_->send(2, z_ind_perm.data(), vec_size * sizeof(Ring));
                network_->send(2, z_label_perm.data(), vec_size * sizeof(Ring));
            }
            network_->flush(2);
            
        } else {
            // Parties 2 to nP: send z values to party 1 for all gates
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                network_->send(1, all_z_key[g].data(), vec_size * sizeof(Ring));
                network_->send(1, all_z_ind[g].data(), vec_size * sizeof(Ring));
                network_->send(1, all_z_label[g].data(), vec_size * sizeof(Ring));
            }
            network_->flush(1);
            
            // Process all gates
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                
                std::vector<Ring> z_key_recv(vec_size);
                std::vector<Ring> z_ind_recv(vec_size);
                std::vector<Ring> z_label_recv(vec_size);
                
                network_->recv(id_ - 1, z_key_recv.data(), vec_size * sizeof(Ring));
                network_->recv(id_ - 1, z_ind_recv.data(), vec_size * sizeof(Ring));
                network_->recv(id_ - 1, z_label_recv.data(), vec_size * sizeof(Ring));
                usleep(latency_);
                
                std::vector<Ring> z_key_send(vec_size);
                std::vector<Ring> z_ind_send(vec_size);
                std::vector<Ring> z_label_send(vec_size);
                
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_gis[g]->shuffle_pi[i];
                    
                    if (id_ != nP_) {
                        z_key_send[i] = z_key_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                        z_ind_send[i] = z_ind_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                        z_label_send[i] = z_label_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt();
                        
                        all_key_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                        all_ind_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                        all_label_shuffled[g][i] = pre_gis[g]->shuffle_c[i].valueAt();
                    } else {
                        z_key_send[i] = z_key_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt() - pre_gis[g]->shuffle_delta[i];
                        z_ind_send[i] = z_ind_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt() - pre_gis[g]->shuffle_delta[i];
                        z_label_send[i] = z_label_recv[pi_i] + pre_gis[g]->shuffle_b[pi_i].valueAt() - pre_gis[g]->shuffle_delta[i];
                        
                        all_key_shuffled[g][i] = z_key_send[i] + pre_gis[g]->shuffle_c[i].valueAt();
                        all_ind_shuffled[g][i] = z_ind_send[i] + pre_gis[g]->shuffle_c[i].valueAt();
                        all_label_shuffled[g][i] = z_label_send[i] + pre_gis[g]->shuffle_c[i].valueAt();
                    }
                }
                
                if (id_ != nP_) {
                    network_->send(id_ + 1, z_key_send.data(), vec_size * sizeof(Ring));
                    network_->send(id_ + 1, z_ind_send.data(), vec_size * sizeof(Ring));
                    network_->send(id_ + 1, z_label_send.data(), vec_size * sizeof(Ring));
                }
            }
            if (id_ != nP_) {
                network_->flush(id_ + 1);
            }
        }
        
        // Step 4: Reconstruct labels for all gates in batch
        size_t total_labels = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            total_labels += vec_sizes[g];
        }
        
        std::vector<Ring> all_labels_flat;
        all_labels_flat.reserve(total_labels);
        for (size_t g = 0; g < num_gates; ++g) {
            all_labels_flat.insert(all_labels_flat.end(), 
                                  all_label_shuffled[g].begin(), 
                                  all_label_shuffled[g].end());
        }
        
        std::vector<Ring> all_labels_reconstructed(total_labels, 0);
        reconstruct(nP_, id_, network_, all_labels_flat, all_labels_reconstructed, use_pking_, latency_);
        
        // Step 5: Apply public permutation and compute key_c for all gates
        std::vector<std::vector<Ring>> all_key_compacted(num_gates);
        std::vector<std::vector<Ring>> all_ind_compacted(num_gates);
        std::vector<std::vector<Ring>> all_key_c_shares(num_gates);
        
        size_t label_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_key_compacted[g].resize(vec_size, Ring(0));
            all_ind_compacted[g].resize(vec_size, Ring(0));
            all_key_c_shares[g].resize(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                Ring label = all_labels_reconstructed[label_offset + i];
                if (label >= 0 && label < static_cast<Ring>(vec_size)) {
                    size_t idx_perm = static_cast<size_t>(label);
                    all_key_compacted[g][idx_perm] = all_key_shuffled[g][i];
                    all_ind_compacted[g][idx_perm] = all_ind_shuffled[g][i];
                }
            }
            
            // Compute key_c
            all_key_c_shares[g][0] = all_ind_compacted[g][0];
            for (size_t i = 1; i < vec_size; ++i) {
                all_key_c_shares[g][i] = all_ind_compacted[g][i-1] - all_ind_compacted[g][i];
            }
            
            label_offset += vec_size;
        }
        
        // Step 6: Compute ind_diff = key_c * key_compacted for all gates (second multiplication)
        std::vector<Ring> all_u_keymult_shares;
        std::vector<Ring> all_v_keymult_shares;
        all_u_keymult_shares.reserve(total_mults);
        all_v_keymult_shares.reserve(total_mults);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            for (size_t j = 0; j < vec_size; ++j) {
                all_u_keymult_shares.push_back(all_key_c_shares[g][j] - pre_gis[g]->keymult_triple_a[j].valueAt());
                all_v_keymult_shares.push_back(all_key_compacted[g][j] - pre_gis[g]->keymult_triple_b[j].valueAt());
            }
        }
        
        // Reconstruct for second multiplication
        std::vector<Ring> keymult_shares_to_send(2 * total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            keymult_shares_to_send[2*i] = all_u_keymult_shares[i];
            keymult_shares_to_send[2*i + 1] = all_v_keymult_shares[i];
        }
        
        std::vector<Ring> keymult_reconstructed(2 * total_mults, 0);
        reconstruct(nP_, id_, network_, keymult_shares_to_send, keymult_reconstructed, use_pking_, latency_);
        
        std::vector<Ring> u_keymult_reconstructed(total_mults);
        std::vector<Ring> v_keymult_reconstructed(total_mults);
        for (size_t i = 0; i < total_mults; ++i) {
            u_keymult_reconstructed[i] = keymult_reconstructed[2*i];
            v_keymult_reconstructed[i] = keymult_reconstructed[2*i + 1];
        }
        
        // Compute ind_diff for all gates
        std::vector<std::vector<Ring>> all_ind_diff_shares(num_gates);
        
        mult_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_ind_diff_shares[g].resize(vec_size);
            
            for (size_t j = 0; j < vec_size; ++j) {
                Ring u = u_keymult_reconstructed[mult_offset + j];
                Ring v = v_keymult_reconstructed[mult_offset + j];
                Ring a = pre_gis[g]->keymult_triple_a[j].valueAt();
                Ring b = pre_gis[g]->keymult_triple_b[j].valueAt();
                Ring c = pre_gis[g]->keymult_triple_c[j].valueAt();
                
                if (id_ == 1) {
                    all_ind_diff_shares[g][j] = u * v + u * b + v * a + c;
                } else {
                    all_ind_diff_shares[g][j] = u * b + v * a + c;
                }
            }
            mult_offset += vec_size;
        }
        
        // Step 7: Apply inverse permutation for all gates
        std::vector<std::vector<Ring>> all_ind_diff_restored(num_gates);
        
        label_offset = 0;
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_ind_diff_restored[g].resize(vec_size, Ring(0));
            
            for (size_t i = 0; i < vec_size; ++i) {
                int label = static_cast<int>(all_labels_reconstructed[label_offset + i]);
                if (label >= 0 && label < static_cast<int>(vec_size)) {
                    all_ind_diff_restored[g][i] = all_ind_diff_shares[g][label];
                }
            }
            label_offset += vec_size;
        }
        
        // Step 8: Second shuffle (reverse compaction) for all gates
        std::vector<std::vector<Ring>> all_ind_diff_final(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_ind_diff_final[g].resize(vec_size);
        }
        
        // Compute z values for second shuffle
        std::vector<std::vector<Ring>> all_z_final(num_gates);
        
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            all_z_final[g].resize(vec_size);
            
            for (size_t i = 0; i < vec_size; ++i) {
                all_z_final[g][i] = all_ind_diff_restored[g][i] + pre_gis[g]->revcompact_shuffle_a[i].valueAt();
            }
        }
        
        if (id_ == 1) {
            // Party 1: collect, permute, and send
            std::vector<std::vector<std::vector<Ring>>> all_z_final_recv(num_gates, std::vector<std::vector<Ring>>(nP_));
            
            for (size_t g = 0; g < num_gates; ++g) {
                all_z_final_recv[g][0] = all_z_final[g];
            }
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                for (size_t g = 0; g < num_gates; ++g) {
                    size_t vec_size = vec_sizes[g];
                    all_z_final_recv[g][pid - 1].resize(vec_size);
                    network_->recv(pid, all_z_final_recv[g][pid - 1].data(), vec_size * sizeof(Ring));
                }
            }
            usleep(latency_);
            
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                
                std::vector<Ring> z_final_sum(vec_size, 0);
                for (int pid = 0; pid < nP_; ++pid) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_final_sum[i] += all_z_final_recv[g][pid][i];
                    }
                }
                
                std::vector<Ring> z_final_perm(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_gis[g]->revcompact_shuffle_pi[i];
                    z_final_perm[i] = z_final_sum[pi_i] + pre_gis[g]->revcompact_shuffle_b[pi_i].valueAt();
                    all_ind_diff_final[g][i] = pre_gis[g]->revcompact_shuffle_c[i].valueAt();
                }
                
                network_->send(2, z_final_perm.data(), vec_size * sizeof(Ring));
            }
            network_->flush(2);
            
        } else {
            // Send to party 1
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                network_->send(1, all_z_final[g].data(), vec_size * sizeof(Ring));
            }
            network_->flush(1);
            
            // Process each gate
            for (size_t g = 0; g < num_gates; ++g) {
                size_t vec_size = vec_sizes[g];
                
                std::vector<Ring> z_final_recv(vec_size);
                network_->recv(id_ - 1, z_final_recv.data(), vec_size * sizeof(Ring));
                usleep(latency_);
                
                std::vector<Ring> z_final_send(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_gis[g]->revcompact_shuffle_pi[i];
                    
                    if (id_ != nP_) {
                        z_final_send[i] = z_final_recv[pi_i] + pre_gis[g]->revcompact_shuffle_b[pi_i].valueAt();
                        all_ind_diff_final[g][i] = pre_gis[g]->revcompact_shuffle_c[i].valueAt();
                    } else {
                        z_final_send[i] = z_final_recv[pi_i] + pre_gis[g]->revcompact_shuffle_b[pi_i].valueAt() - pre_gis[g]->revcompact_shuffle_delta[i];
                        all_ind_diff_final[g][i] = z_final_send[i] + pre_gis[g]->revcompact_shuffle_c[i].valueAt();
                    }
                }
                
                if (id_ != nP_) {
                    network_->send(id_ + 1, z_final_send.data(), vec_size * sizeof(Ring));
                }
            }
            if (id_ != nP_) {
                network_->flush(id_ + 1);
            }
        }
        
        // Step 9: Compute final group indices via prefix sum and write outputs
        for (size_t g = 0; g < num_gates; ++g) {
            size_t vec_size = vec_sizes[g];
            std::vector<Ring> group_index_shares(vec_size);
            
            Ring running_sum = 0;
            for (size_t i = 0; i < vec_size; ++i) {
                running_sum += all_ind_diff_final[g][i];
                if (id_ == 1) {
                    group_index_shares[i] = running_sum + Ring(i);
                } else {
                    group_index_shares[i] = running_sum;
                }
            }
            
            // Write outputs
            for (size_t i = 0; i < vec_size; ++i) {
                wires_[gi_gates[g].outs[i]] = group_index_shares[i];
                wires_[gi_gates[g].outs[vec_size + i]] = all_key_shares[g][i];
                wires_[gi_gates[g].outs[2 * vec_size + i]] = all_v_shares[g][i];
            }
        }
    }

    void OnlineEvaluator::groupwisePropagateEvaluate(const common::utils::SIMDOGate &gp_gate, int latency) {
        if (id_ == 0) { return; }
        
        auto *pre_gp = static_cast<PreprocGroupwisePropagateGate<Ring> *>(preproc_.gates[gp_gate.out].get());
        
        // Input format: [key1_0,...,key1_n1, v1_0,...,v1_n1, key2_0,...,key2_n2]
        // Output format: [key2_0,...,key2_n2, v_out_0,...,v_out_n2]
        
        // Determine sizes from preprocessing or gate metadata
        size_t t1_vec_size = pre_gp->t1_shuffle_a.size();
        size_t t2_vec_size = pre_gp->t2_shuffle_a.size();
        
        // Extract input vectors
        std::vector<Ring> key1_shares(t1_vec_size);
        std::vector<Ring> v1_shares(t1_vec_size);
        std::vector<Ring> key2_shares(t2_vec_size);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            key1_shares[i] = wires_[gp_gate.in[i]];
            v1_shares[i] = wires_[gp_gate.in[t1_vec_size + i]];
        }
        for (size_t i = 0; i < t2_vec_size; ++i) {
            key2_shares[i] = wires_[gp_gate.in[2 * t1_vec_size + i]];
        }
        
        // Step 1 & 2: Compact T1 and T2 in parallel
        
        // Step 1a & 2a: Compute prefix sums c0 and c1 locally on key1 and key2 shares
        std::vector<Ring> c1_t1_shares(t1_vec_size);
        std::vector<Ring> c0_t1_shares(t1_vec_size);
        std::vector<Ring> c1_t2_shares(t2_vec_size);
        std::vector<Ring> c0_t2_shares(t2_vec_size);
        
        // T1 prefix sums
        c1_t1_shares[0] = key1_shares[0];
        if (id_ == 1) {
            c0_t1_shares[0] = Ring(1) - key1_shares[0];
        } else {
            c0_t1_shares[0] = Ring(0) - key1_shares[0];
        }
        
        for (size_t j = 1; j < t1_vec_size; ++j) {
            c1_t1_shares[j] = c1_t1_shares[j - 1] + key1_shares[j];
            c0_t1_shares[j] = c0_t1_shares[j - 1] + (id_ == 1 ? Ring(1) : Ring(0)) - key1_shares[j];
        }
        
        // T2 prefix sums
        c1_t2_shares[0] = key2_shares[0];
        if (id_ == 1) {
            c0_t2_shares[0] = Ring(1) - key2_shares[0];
        } else {
            c0_t2_shares[0] = Ring(0) - key2_shares[0];
        }
        
        for (size_t j = 1; j < t2_vec_size; ++j) {
            c1_t2_shares[j] = c1_t2_shares[j - 1] + key2_shares[j];
            c0_t2_shares[j] = c0_t2_shares[j - 1] + (id_ == 1 ? Ring(1) : Ring(0)) - key2_shares[j];
        }
        
        // Step 1b & 2b: Compute label shares for T1 and T2
        // label[j] = (c0[j] + c1[N-1] - c1[j])(1 - t[j]) + c1[j] - 1
        std::vector<Ring> label_t1_shares(t1_vec_size);
        std::vector<Ring> label_t2_shares(t2_vec_size);
        Ring c1_t1_last = c1_t1_shares[t1_vec_size - 1];
        Ring c1_t2_last = c1_t2_shares[t2_vec_size - 1];
        
        // Prepare masked values for reconstruction using Beaver triples
        int pKing = 1;
        std::vector<Ring> u_t1_shares(t1_vec_size);  // diff_term - a
        std::vector<Ring> v_t1_shares(t1_vec_size);  // (1-t) - b
        std::vector<Ring> u_t2_shares(t2_vec_size);
        std::vector<Ring> v_t2_shares(t2_vec_size);
        
        for (size_t j = 0; j < t1_vec_size; ++j) {
            Ring diff_term = c0_t1_shares[j] + c1_t1_last - c1_t1_shares[j];
            Ring one_minus_t;
            if (id_ == 1) {
                one_minus_t = Ring(1) - key1_shares[j];
            } else {
                one_minus_t = Ring(0) - key1_shares[j];
            }
            
            u_t1_shares[j] = diff_term - pre_gp->t1_mult_triple_a[j].valueAt();
            v_t1_shares[j] = one_minus_t - pre_gp->t1_mult_triple_b[j].valueAt();
        }
        
        for (size_t j = 0; j < t2_vec_size; ++j) {
            Ring diff_term = c0_t2_shares[j] + c1_t2_last - c1_t2_shares[j];
            Ring one_minus_t;
            if (id_ == 1) {
                one_minus_t = Ring(1) - key2_shares[j];
            } else {
                one_minus_t = Ring(0) - key2_shares[j];
            }
            
            u_t2_shares[j] = diff_term - pre_gp->t2_mult_triple_a[j].valueAt();
            v_t2_shares[j] = one_minus_t - pre_gp->t2_mult_triple_b[j].valueAt();
        }
        
        // Step 1c & 2c: Reconstruct u and v for T1 and T2 together
        std::vector<Ring> u_t1_reconstructed(t1_vec_size, 0);
        std::vector<Ring> v_t1_reconstructed(t1_vec_size, 0);
        std::vector<Ring> u_t2_reconstructed(t2_vec_size, 0);
        std::vector<Ring> v_t2_reconstructed(t2_vec_size, 0);
        
        std::vector<Ring> shares_to_send(2 * t1_vec_size + 2 * t2_vec_size);
        for (size_t i = 0; i < t1_vec_size; ++i) {
            shares_to_send[2*i] = u_t1_shares[i];
            shares_to_send[2*i + 1] = v_t1_shares[i];
        }
        for (size_t i = 0; i < t2_vec_size; ++i) {
            shares_to_send[2 * t1_vec_size + 2*i] = u_t2_shares[i];
            shares_to_send[2 * t1_vec_size + 2*i + 1] = v_t2_shares[i];
        }

        // King-based reconstruction
        std::vector<Ring> reconstructed(2 * t1_vec_size + 2 * t2_vec_size, 0);
        reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            u_t1_reconstructed[i] = reconstructed[2*i];
            v_t1_reconstructed[i] = reconstructed[2*i + 1];
        }
        for (size_t i = 0; i < t2_vec_size; ++i) {
            u_t2_reconstructed[i] = reconstructed[2 * t1_vec_size + 2*i];
            v_t2_reconstructed[i] = reconstructed[2 * t1_vec_size + 2*i + 1];
        }
        
        // Step 1d & 2d: Compute label shares for T1 and T2
        for (size_t j = 0; j < t1_vec_size; ++j) {
            Ring u = u_t1_reconstructed[j];
            Ring v = v_t1_reconstructed[j];
            Ring a = pre_gp->t1_mult_triple_a[j].valueAt();
            Ring b = pre_gp->t1_mult_triple_b[j].valueAt();
            Ring c = pre_gp->t1_mult_triple_c[j].valueAt();
            
            if (id_ == 1) {
                label_t1_shares[j] = u * v + u * b + v * a + c + c1_t1_shares[j] - Ring(1);
            } else {
                label_t1_shares[j] = u * b + v * a + c + c1_t1_shares[j];
            }
        }
        
        for (size_t j = 0; j < t2_vec_size; ++j) {
            Ring u = u_t2_reconstructed[j];
            Ring v = v_t2_reconstructed[j];
            Ring a = pre_gp->t2_mult_triple_a[j].valueAt();
            Ring b = pre_gp->t2_mult_triple_b[j].valueAt();
            Ring c = pre_gp->t2_mult_triple_c[j].valueAt();
            
            if (id_ == 1) {
                label_t2_shares[j] = u * v + u * b + v * a + c + c1_t2_shares[j] - Ring(1);
            } else {
                label_t2_shares[j] = u * b + v * a + c + c1_t2_shares[j];
            }
        }
        
        // Step 1e & 2e: Shuffle T1 (key1, v1, label_t1) and T2 (key2, label_t2) together
        std::vector<Ring> key1_shuffled(t1_vec_size);
        std::vector<Ring> v1_shuffled(t1_vec_size);
        std::vector<Ring> label_t1_shuffled(t1_vec_size);
        std::vector<Ring> key2_shuffled(t2_vec_size);
        std::vector<Ring> label_t2_shuffled(t2_vec_size);
        
        // Compute z = input + a for each vector
        std::vector<Ring> z_key1(t1_vec_size);
        std::vector<Ring> z_v1(t1_vec_size);
        std::vector<Ring> z_label_t1(t1_vec_size);
        std::vector<Ring> z_key2(t2_vec_size);
        std::vector<Ring> z_label_t2(t2_vec_size);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            z_key1[i] = key1_shares[i] + pre_gp->t1_shuffle_a[i].valueAt();
            z_v1[i] = v1_shares[i] + pre_gp->t1_shuffle_a[i].valueAt();
            z_label_t1[i] = label_t1_shares[i] + pre_gp->t1_shuffle_a[i].valueAt();
        }
        
        for (size_t i = 0; i < t2_vec_size; ++i) {
            z_key2[i] = key2_shares[i] + pre_gp->t2_shuffle_a[i].valueAt();
            z_label_t2[i] = label_t2_shares[i] + pre_gp->t2_shuffle_a[i].valueAt();
        }
        
        if (id_ == 1) {
            // Party 1: Collect z from all parties, apply permutation, send to party 2
            std::vector<std::vector<Ring>> z_key1_recv(nP_);
            std::vector<std::vector<Ring>> z_v1_recv(nP_);
            std::vector<std::vector<Ring>> z_label_t1_recv(nP_);
            std::vector<std::vector<Ring>> z_key2_recv(nP_);
            std::vector<std::vector<Ring>> z_label_t2_recv(nP_);
            
            z_key1_recv[0] = z_key1;
            z_v1_recv[0] = z_v1;
            z_label_t1_recv[0] = z_label_t1;
            z_key2_recv[0] = z_key2;
            z_label_t2_recv[0] = z_label_t2;
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_key1_recv[pid - 1].resize(t1_vec_size);
                z_v1_recv[pid - 1].resize(t1_vec_size);
                z_label_t1_recv[pid - 1].resize(t1_vec_size);
                z_key2_recv[pid - 1].resize(t2_vec_size);
                z_label_t2_recv[pid - 1].resize(t2_vec_size);
                
                network_->recv(pid, z_key1_recv[pid - 1].data(), t1_vec_size * sizeof(Ring));
                network_->recv(pid, z_v1_recv[pid - 1].data(), t1_vec_size * sizeof(Ring));
                network_->recv(pid, z_label_t1_recv[pid - 1].data(), t1_vec_size * sizeof(Ring));
                network_->recv(pid, z_key2_recv[pid - 1].data(), t2_vec_size * sizeof(Ring));
                network_->recv(pid, z_label_t2_recv[pid - 1].data(), t2_vec_size * sizeof(Ring));
            }
            usleep(latency);
            
            // Sum all z values
            std::vector<Ring> z_key1_sum(t1_vec_size, 0);
            std::vector<Ring> z_v1_sum(t1_vec_size, 0);
            std::vector<Ring> z_label_t1_sum(t1_vec_size, 0);
            std::vector<Ring> z_key2_sum(t2_vec_size, 0);
            std::vector<Ring> z_label_t2_sum(t2_vec_size, 0);
            
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < t1_vec_size; ++i) {
                    z_key1_sum[i] += z_key1_recv[pid][i];
                    z_v1_sum[i] += z_v1_recv[pid][i];
                    z_label_t1_sum[i] += z_label_t1_recv[pid][i];
                }
                for (size_t i = 0; i < t2_vec_size; ++i) {
                    z_key2_sum[i] += z_key2_recv[pid][i];
                    z_label_t2_sum[i] += z_label_t2_recv[pid][i];
                }
            }
            
            // Apply permutation and add b
            std::vector<Ring> z_key1_perm(t1_vec_size);
            std::vector<Ring> z_v1_perm(t1_vec_size);
            std::vector<Ring> z_label_t1_perm(t1_vec_size);
            std::vector<Ring> z_key2_perm(t2_vec_size);
            std::vector<Ring> z_label_t2_perm(t2_vec_size);
            
            for (size_t i = 0; i < t1_vec_size; ++i) {
                int pi_i = pre_gp->t1_shuffle_pi[i];
                z_key1_perm[i] = z_key1_sum[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                z_v1_perm[i] = z_v1_sum[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                z_label_t1_perm[i] = z_label_t1_sum[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                
                key1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
                v1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
                label_t1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
            }
            
            for (size_t i = 0; i < t2_vec_size; ++i) {
                int pi_i = pre_gp->t2_shuffle_pi[i];
                z_key2_perm[i] = z_key2_sum[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt();
                z_label_t2_perm[i] = z_label_t2_sum[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt();
                
                key2_shuffled[i] = pre_gp->t2_shuffle_c[i].valueAt();
                label_t2_shuffled[i] = pre_gp->t2_shuffle_c[i].valueAt();
            }
            
            // Send to party 2
            network_->send(2, z_key1_perm.data(), t1_vec_size * sizeof(Ring));
            network_->send(2, z_v1_perm.data(), t1_vec_size * sizeof(Ring));
            network_->send(2, z_label_t1_perm.data(), t1_vec_size * sizeof(Ring));
            network_->send(2, z_key2_perm.data(), t2_vec_size * sizeof(Ring));
            network_->send(2, z_label_t2_perm.data(), t2_vec_size * sizeof(Ring));
            network_->flush(2);
            
        } else {
            // Parties 2 to nP: Send z to party 1
            network_->send(1, z_key1.data(), t1_vec_size * sizeof(Ring));
            network_->send(1, z_v1.data(), t1_vec_size * sizeof(Ring));
            network_->send(1, z_label_t1.data(), t1_vec_size * sizeof(Ring));
            network_->send(1, z_key2.data(), t2_vec_size * sizeof(Ring));
            network_->send(1, z_label_t2.data(), t2_vec_size * sizeof(Ring));
            network_->flush(1);
            
            // Receive from previous party
            std::vector<Ring> z_key1_recv(t1_vec_size);
            std::vector<Ring> z_v1_recv(t1_vec_size);
            std::vector<Ring> z_label_t1_recv(t1_vec_size);
            std::vector<Ring> z_key2_recv(t2_vec_size);
            std::vector<Ring> z_label_t2_recv(t2_vec_size);
            
            network_->recv(id_ - 1, z_key1_recv.data(), t1_vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_v1_recv.data(), t1_vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_label_t1_recv.data(), t1_vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_key2_recv.data(), t2_vec_size * sizeof(Ring));
            network_->recv(id_ - 1, z_label_t2_recv.data(), t2_vec_size * sizeof(Ring));
            usleep(latency);
            
            // Apply permutation and add b
            std::vector<Ring> z_key1_send(t1_vec_size);
            std::vector<Ring> z_v1_send(t1_vec_size);
            std::vector<Ring> z_label_t1_send(t1_vec_size);
            std::vector<Ring> z_key2_send(t2_vec_size);
            std::vector<Ring> z_label_t2_send(t2_vec_size);
            
            for (size_t i = 0; i < t1_vec_size; ++i) {
                int pi_i = pre_gp->t1_shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_key1_send[i] = z_key1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                    z_v1_send[i] = z_v1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                    z_label_t1_send[i] = z_label_t1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt();
                    
                    key1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
                    v1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
                    label_t1_shuffled[i] = pre_gp->t1_shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_key1_send[i] = z_key1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt() - pre_gp->t1_shuffle_delta[i];
                    z_v1_send[i] = z_v1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt() - pre_gp->t1_shuffle_delta[i];
                    z_label_t1_send[i] = z_label_t1_recv[pi_i] + pre_gp->t1_shuffle_b[pi_i].valueAt() - pre_gp->t1_shuffle_delta[i];
                    
                    key1_shuffled[i] = z_key1_send[i] + pre_gp->t1_shuffle_c[i].valueAt();
                    v1_shuffled[i] = z_v1_send[i] + pre_gp->t1_shuffle_c[i].valueAt();
                    label_t1_shuffled[i] = z_label_t1_send[i] + pre_gp->t1_shuffle_c[i].valueAt();
                }
            }
            
            for (size_t i = 0; i < t2_vec_size; ++i) {
                int pi_i = pre_gp->t2_shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_key2_send[i] = z_key2_recv[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt();
                    z_label_t2_send[i] = z_label_t2_recv[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt();
                    
                    key2_shuffled[i] = pre_gp->t2_shuffle_c[i].valueAt();
                    label_t2_shuffled[i] = pre_gp->t2_shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_key2_send[i] = z_key2_recv[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt() - pre_gp->t2_shuffle_delta[i];
                    z_label_t2_send[i] = z_label_t2_recv[pi_i] + pre_gp->t2_shuffle_b[pi_i].valueAt() - pre_gp->t2_shuffle_delta[i];
                    
                    key2_shuffled[i] = z_key2_send[i] + pre_gp->t2_shuffle_c[i].valueAt();
                    label_t2_shuffled[i] = z_label_t2_send[i] + pre_gp->t2_shuffle_c[i].valueAt();
                }
            }
            
            // Send to next party (if not last)
            if (id_ != nP_) {
                network_->send(id_ + 1, z_key1_send.data(), t1_vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_v1_send.data(), t1_vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_label_t1_send.data(), t1_vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_key2_send.data(), t2_vec_size * sizeof(Ring));
                network_->send(id_ + 1, z_label_t2_send.data(), t2_vec_size * sizeof(Ring));
                network_->flush(id_ + 1);
            }
        }
        
        
        // Step 1f & 2f: Reconstruct label_t1_shuffled and label_t2_shuffled together
        std::vector<Ring> label_t1_reconstructed(t1_vec_size, 0);
        std::vector<Ring> label_t2_reconstructed(t2_vec_size, 0);
        
        std::vector<Ring> labels_to_send(t1_vec_size + t2_vec_size);
        for (size_t i = 0; i < t1_vec_size; ++i) {
            labels_to_send[i] = label_t1_shuffled[i];
        }
        for (size_t i = 0; i < t2_vec_size; ++i) {
            labels_to_send[t1_vec_size + i] = label_t2_shuffled[i];
        }
        
        std::vector<Ring> labels_reconstructed(t1_vec_size + t2_vec_size, 0);
        reconstruct(nP_, id_, network_, labels_to_send, labels_reconstructed, use_pking_, latency);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            label_t1_reconstructed[i] = labels_reconstructed[i];
        }
        for (size_t i = 0; i < t2_vec_size; ++i) {
            label_t2_reconstructed[i] = labels_reconstructed[t1_vec_size + i];
        }
        
        // Step 1g & 2g: Apply public permutation based on reconstructed labels
        std::vector<Ring> key1_compacted(t1_vec_size);
        std::vector<Ring> v1_compacted(t1_vec_size);
        std::vector<Ring> key2_compacted(t2_vec_size);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            Ring label = label_t1_reconstructed[i];
            if (label >= 0 && label < static_cast<Ring>(t1_vec_size)) {
                size_t idx = static_cast<size_t>(label);
                key1_compacted[idx] = key1_shuffled[i];
                v1_compacted[idx] = v1_shuffled[i];
            }
        }
        
        for (size_t i = 0; i < t2_vec_size; ++i) {
            Ring label = label_t2_reconstructed[i];
            if (label >= 0 && label < static_cast<Ring>(t2_vec_size)) {
                size_t idx = static_cast<size_t>(label);
                key2_compacted[idx] = key2_shuffled[i];
            }
        }

        // Step 3: Compute differences and multiply by keys
        // diff[i] = v1_compacted[i] - v1_compacted[i-1] for i > 0, diff[0] = v1_compacted[0]
        // result[i] = diff[i] * key1_compacted[i]
        
        std::vector<Ring> diff_shares(t1_vec_size);
        if (id_ == 1) {
            diff_shares[0] = v1_compacted[0];
        } else {
            diff_shares[0] = v1_compacted[0];
        }
        
        for (size_t i = 1; i < t1_vec_size; ++i) {
            diff_shares[i] = v1_compacted[i] - v1_compacted[i - 1];
        }
        
        // Step 3b: Multiply diff[i] * key1_compacted[i] using Beaver triples
        // Prepare masked values for reconstruction
        std::vector<Ring> u_diff_shares(t1_vec_size);  // diff - a
        std::vector<Ring> v_key_shares(t1_vec_size);  // key - b
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            u_diff_shares[i] = diff_shares[i] - pre_gp->diff_mult_triple_a[i].valueAt();
            v_key_shares[i] = key1_compacted[i] - pre_gp->diff_mult_triple_b[i].valueAt();
        }
        
        // Step 3c: Reconstruct u_diff and v_key
        std::vector<Ring> u_diff_reconstructed(t1_vec_size, 0);
        std::vector<Ring> v_key_reconstructed(t1_vec_size, 0);
        
        std::vector<Ring> shares_to_send_mult(2 * t1_vec_size);
        for (size_t i = 0; i < t1_vec_size; ++i) {
            shares_to_send_mult[2*i] = u_diff_shares[i];
            shares_to_send_mult[2*i + 1] = v_key_shares[i];
        }
        
        std::vector<Ring> reconstructed_mult(2 * t1_vec_size, 0);
        reconstruct(nP_, id_, network_, shares_to_send_mult, reconstructed_mult, use_pking_, latency);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            u_diff_reconstructed[i] = reconstructed_mult[2*i];
            v_key_reconstructed[i] = reconstructed_mult[2*i + 1];
        }
        
        // Step 3d: Compute multiplication result shares
        std::vector<Ring> key_times_diff_shares(t1_vec_size);
        
        for (size_t i = 0; i < t1_vec_size; ++i) {
            Ring u = u_diff_reconstructed[i];
            Ring v = v_key_reconstructed[i];
            Ring a = pre_gp->diff_mult_triple_a[i].valueAt();
            Ring b = pre_gp->diff_mult_triple_b[i].valueAt();
            Ring c = pre_gp->diff_mult_triple_c[i].valueAt();
            
            if (id_ == 1) {
                key_times_diff_shares[i] = u * v + u * b + v * a + c;
            } else {
                key_times_diff_shares[i] = u * b + v * a + c;
            }
        }
        
        
        // Step 4a: Create extended vector matching T2 size
        // Since T1 is compacted and has size t1_vec_size, but T2 has size t2_vec_size,
        // we pad the key_times_diff to T2 size
        std::vector<Ring> extended_diff_shares_(t2_vec_size);
        for (size_t i = 0; i < t2_vec_size; ++i) {
            if (i < t1_vec_size) {
                extended_diff_shares_[i] = key_times_diff_shares[i];
            } else {
                extended_diff_shares_[i] = Ring(0);
            }
        }
        
        // Step 4b-pre: Reverse the public permutation applied in Step 2g

        std::vector<Ring> key2_uncompacted(t2_vec_size);
        
        for (size_t i = 0; i < t2_vec_size; ++i) {
            Ring label = label_t2_reconstructed[i];
            if (label >= 0 && label < static_cast<Ring>(t2_vec_size)) {
                size_t idx = static_cast<size_t>(label);
                key2_uncompacted[i] = key2_compacted[idx];
            } else {
                key2_uncompacted[i] = Ring(0);
            }
        }
        
        // Step 4b-pre: Reverse the public permutation applied in Step 2g
        std::vector<Ring> extended_diff_shares(t2_vec_size);
        
        for (size_t i = 0; i < t2_vec_size; ++i) {
            Ring label = label_t2_reconstructed[i];
            if (label >= 0 && label < static_cast<Ring>(t2_vec_size)) {
                size_t idx = static_cast<size_t>(label);
                extended_diff_shares[i] = extended_diff_shares_[idx];
            } else {
                extended_diff_shares[i] = Ring(0);
            }
        }
        
        // Step 4b: Shuffle extended_diff back using reverse compaction shuffle
        std::vector<Ring> diff_reverse_shuffled(t2_vec_size);
        
        // Compute z = input + a for reverse compaction
        std::vector<Ring> z_diff_rev(t2_vec_size);
        
        for (size_t i = 0; i < t2_vec_size; ++i) {
            z_diff_rev[i] = extended_diff_shares[i] + pre_gp->revcompact_shuffle_a[i].valueAt();
        }
        
        if (id_ == 1) {
            // Party 1: Collect z from all parties, apply permutation, send to party 2
            std::vector<std::vector<Ring>> z_diff_rev_recv(nP_);
            z_diff_rev_recv[0] = z_diff_rev;
            
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_diff_rev_recv[pid - 1].resize(t2_vec_size);
                network_->recv(pid, z_diff_rev_recv[pid - 1].data(), t2_vec_size * sizeof(Ring));
            }
            usleep(latency);
            
            // Sum all z values
            std::vector<Ring> z_diff_rev_sum(t2_vec_size, 0);
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < t2_vec_size; ++i) {
                    z_diff_rev_sum[i] += z_diff_rev_recv[pid][i];
                }
            }
            
            // Apply permutation and add b
            std::vector<Ring> z_diff_rev_perm(t2_vec_size);
            for (size_t i = 0; i < t2_vec_size; ++i) {
                int pi_i = pre_gp->revcompact_shuffle_pi[i];
                z_diff_rev_perm[i] = z_diff_rev_sum[pi_i] + pre_gp->revcompact_shuffle_b[pi_i].valueAt();
                diff_reverse_shuffled[i] = pre_gp->revcompact_shuffle_c[i].valueAt();
            }
            
            // Send to party 2
            network_->send(2, z_diff_rev_perm.data(), t2_vec_size * sizeof(Ring));
            network_->flush(2);
            
        } else {
            // Parties 2 to nP: Send z to party 1
            network_->send(1, z_diff_rev.data(), t2_vec_size * sizeof(Ring));
            network_->flush(1);
            
            // Receive from previous party
            std::vector<Ring> z_diff_rev_recv(t2_vec_size);
            network_->recv(id_ - 1, z_diff_rev_recv.data(), t2_vec_size * sizeof(Ring));
            usleep(latency);
            
            // Apply permutation and add b
            std::vector<Ring> z_diff_rev_send(t2_vec_size);
            for (size_t i = 0; i < t2_vec_size; ++i) {
                int pi_i = pre_gp->revcompact_shuffle_pi[i];
                
                if (id_ != nP_) {
                    z_diff_rev_send[i] = z_diff_rev_recv[pi_i] + pre_gp->revcompact_shuffle_b[pi_i].valueAt();
                    diff_reverse_shuffled[i] = pre_gp->revcompact_shuffle_c[i].valueAt();
                } else {
                    // Last party subtracts delta
                    z_diff_rev_send[i] = z_diff_rev_recv[pi_i] + pre_gp->revcompact_shuffle_b[pi_i].valueAt() - pre_gp->revcompact_shuffle_delta[i];
                    diff_reverse_shuffled[i] = z_diff_rev_send[i] + pre_gp->revcompact_shuffle_c[i].valueAt();
                }
            }
            
            // Send to next party (if not last)
            if (id_ != nP_) {
                network_->send(id_ + 1, z_diff_rev_send.data(), t2_vec_size * sizeof(Ring));
                network_->flush(id_ + 1);
            }
        }

        // Step 5: Prefix sum to propagate values within groups
        // This computes cumulative sums from left to right
        // result[i] = diff_reverse_shuffled[0] + diff_reverse_shuffled[1] + ... + diff_reverse_shuffled[i]
        
        std::vector<Ring> propagated_values(t2_vec_size);
        
        // Start from the first element and work forward
        propagated_values[0] = diff_reverse_shuffled[0];
        
        for (size_t i = 1; i < t2_vec_size; ++i) {
            propagated_values[i] = propagated_values[i - 1] + diff_reverse_shuffled[i];
        }

        // Final output: key2_uncompacted (restored key2) and propagated_values
        for (size_t i = 0; i < t2_vec_size; ++i) {
            wires_[gp_gate.outs[i]] = key2_uncompacted[i];
            wires_[gp_gate.outs[t2_vec_size + i]] = propagated_values[i];
        }
    }

    void OnlineEvaluator::groupwisePropagateEvaluateParallel(const std::vector<common::utils::SIMDOGate> &gp_gates) {
        if (id_ == 0) { return; }

        size_t num_gp_gates = gp_gates.size();
        if (num_gp_gates == 0) { return; }

        // Process each GP gate in parallel
        // #pragma omp parallel for
        for (size_t i = 0; i < num_gp_gates; ++i) {
            if(i == 0){
                groupwisePropagateEvaluate(gp_gates[i], latency_);
            } else {
                groupwisePropagateEvaluate(gp_gates[i], 0);
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

    void OnlineEvaluator::sortEvaluate(const std::vector<common::utils::SIMDOGate> &sort_gates) {
        if (id_ == 0 || sort_gates.empty()) { return; }
        
        for (const auto& sort_gate : sort_gates) {
            auto *pre_sort = static_cast<PreprocSortGate<Ring> *>(preproc_.gates[sort_gate.out].get());
            
            // Input: [bit0_elem0, bit1_elem0, ..., bit31_elem0, bit0_elem1, ...]
            size_t total_size = sort_gate.in.size();
            size_t vec_size = total_size / 32;
            
            if (vec_size == 0 || total_size % 32 != 0) {
                throw std::runtime_error("Sort gate input size must be divisible by 32");
            }
            
            // Initialize payload as identity permutation [0, 1, 2, ..., vec_size-1] (0-indexed)
            std::vector<Ring> payload_shares(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                if (id_ == 1) {
                    payload_shares[i] = static_cast<Ring>(i);
                } else {
                    payload_shares[i] = Ring(0);
                }
            }
            
            // Storage for bits being processed
            // bits_current[i] = shares of bit i for all elements (after compaction)
            std::vector<std::vector<Ring>> bits_current(32, std::vector<Ring>(vec_size));
            
            // Initialize: extract bit-decomposed input
            // Input format: consecutive 32 bits per element
            for (size_t elem = 0; elem < vec_size; ++elem) {
                for (size_t bit = 0; bit < 32; ++bit) {
                    bits_current[bit][elem] = wires_[sort_gate.in[elem * 32 + bit]];
                }
            }
            
            // Main loop: process each bit from MSB (bit 31) to LSB (bit 0)
            // In our storage: bit 31 is MSB, bit 0 is LSB
            for (int bit_idx = 31; bit_idx >= 0; --bit_idx) {
                // For preprocessing, we use array index (0 = first iteration = MSB)
                int preproc_idx = 31 - bit_idx;
                
                // Extract current bit as key vector
                std::vector<Ring> key_shares = bits_current[bit_idx];
                
                // Prepare payloads: all higher bits + current payload
                std::vector<std::vector<Ring>> compact_payloads;
                for (int higher_bit = 31; higher_bit > bit_idx; --higher_bit) {
                    compact_payloads.push_back(bits_current[higher_bit]);
                }
                compact_payloads.push_back(payload_shares);
                
                size_t num_payloads = compact_payloads.size();
                
                // Perform compaction using the preprocessing for this bit
                // This is similar to compactEvaluate but using pre_sort data
                
                // Step 1: Compute prefix sums c0 and c1 locally
                std::vector<Ring> c1_shares(vec_size);
                std::vector<Ring> c0_shares(vec_size);
                
                c1_shares[0] = key_shares[0];
                if (id_ == 1) {
                    c0_shares[0] = Ring(1) - c1_shares[0];
                } else {
                    c0_shares[0] = -c1_shares[0];
                }
                
                for (size_t j = 1; j < vec_size; ++j) {
                    c1_shares[j] = c1_shares[j-1] + key_shares[j];
                    if (id_ == 1) {
                        c0_shares[j] = Ring(j+1) - c1_shares[j];
                    } else {
                        c0_shares[j] = -c1_shares[j];
                    }
                }
                
                // Step 2: Compute label shares using multiplications
                std::vector<Ring> label_shares(vec_size);
                Ring c1_last = c1_shares[vec_size - 1];
                
                std::vector<Ring> u_shares(vec_size);
                std::vector<Ring> v_shares(vec_size);
                
                for (size_t j = 0; j < vec_size; ++j) {
                    Ring diff_term = c0_shares[j] + c1_last - c1_shares[j];
                    Ring one_minus_t;
                    if (id_ == 1) {
                        one_minus_t = Ring(1) - key_shares[j];
                    } else {
                        one_minus_t = -key_shares[j];
                    }
                    
                    u_shares[j] = diff_term - pre_sort->mult_triple_a[preproc_idx][j].valueAt();
                    v_shares[j] = one_minus_t - pre_sort->mult_triple_b[preproc_idx][j].valueAt();
                }
                
                std::vector<Ring> u_reconstructed(vec_size, 0);
                std::vector<Ring> v_reconstructed(vec_size, 0);
                
                std::vector<Ring> shares_to_send(2 * vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    shares_to_send[2*i] = u_shares[i];
                    shares_to_send[2*i + 1] = v_shares[i];
                }
                
                std::vector<Ring> reconstructed(2 * vec_size, 0);
                reconstruct(nP_, id_, network_, shares_to_send, reconstructed, use_pking_, latency_);
                
                for (size_t i = 0; i < vec_size; ++i) {
                    u_reconstructed[i] = reconstructed[2*i];
                    v_reconstructed[i] = reconstructed[2*i + 1];
                }
                
                for (size_t j = 0; j < vec_size; ++j) {
                    Ring u = u_reconstructed[j];
                    Ring v = v_reconstructed[j];
                    Ring a = pre_sort->mult_triple_a[preproc_idx][j].valueAt();
                    Ring b = pre_sort->mult_triple_b[preproc_idx][j].valueAt();
                    Ring c = pre_sort->mult_triple_c[preproc_idx][j].valueAt();
                    
                    Ring mult_result;
                    if (id_ == 1) {
                        mult_result = u * v + u * b + v * a + c;
                        label_shares[j] = mult_result + c1_shares[j] - Ring(1);
                    } else {
                        mult_result = u * b + v * a + c;
                        label_shares[j] = mult_result + c1_shares[j];
                    }
                }
                
                // Step 3: Shuffle key, all payloads, and label
                std::vector<Ring> key_shuffled(vec_size);
                std::vector<std::vector<Ring>> payloads_shuffled(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> label_shuffled(vec_size);
                
                // Compute z = input + a for shuffle
                std::vector<Ring> z_key(vec_size);
                std::vector<std::vector<Ring>> z_payloads(num_payloads, std::vector<Ring>(vec_size));
                std::vector<Ring> z_label(vec_size);
                
                for (size_t i = 0; i < vec_size; ++i) {
                    z_key[i] = key_shares[i] + pre_sort->shuffle_a[preproc_idx][i].valueAt();
                    z_label[i] = label_shares[i] + pre_sort->shuffle_a[preproc_idx][i].valueAt();
                }
                for (size_t p = 0; p < num_payloads; ++p) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_payloads[p][i] = compact_payloads[p][i] + pre_sort->shuffle_a[preproc_idx][i].valueAt();
                    }
                }
                
                // Shuffle protocol (same as in compactEvaluate)
                if (id_ == 1) {
                    std::vector<std::vector<Ring>> z_key_recv(nP_);
                    std::vector<std::vector<std::vector<Ring>>> z_payloads_recv(nP_, std::vector<std::vector<Ring>>(num_payloads));
                    std::vector<std::vector<Ring>> z_label_recv(nP_);
                    
                    z_key_recv[0] = z_key;
                    z_payloads_recv[0] = z_payloads;
                    z_label_recv[0] = z_label;
                    
                    #pragma omp parallel for
                    for (int pid = 2; pid <= nP_; ++pid) {
                        z_key_recv[pid - 1].resize(vec_size);
                        z_label_recv[pid - 1].resize(vec_size);
                        network_->recv(pid, z_key_recv[pid - 1].data(), vec_size * sizeof(Ring));
                        for (size_t p = 0; p < num_payloads; ++p) {
                            z_payloads_recv[pid - 1][p].resize(vec_size);
                            network_->recv(pid, z_payloads_recv[pid - 1][p].data(), vec_size * sizeof(Ring));
                        }
                        network_->recv(pid, z_label_recv[pid - 1].data(), vec_size * sizeof(Ring));
                    }
                    usleep(latency_);
                    
                    std::vector<Ring> z_key_sum(vec_size, 0);
                    std::vector<std::vector<Ring>> z_payloads_sum(num_payloads, std::vector<Ring>(vec_size, 0));
                    std::vector<Ring> z_label_sum(vec_size, 0);
                    
                    for (int pid = 0; pid < nP_; ++pid) {
                        for (size_t i = 0; i < vec_size; ++i) {
                            z_key_sum[i] += z_key_recv[pid][i];
                            z_label_sum[i] += z_label_recv[pid][i];
                        }
                        for (size_t p = 0; p < num_payloads; ++p) {
                            for (size_t i = 0; i < vec_size; ++i) {
                                z_payloads_sum[p][i] += z_payloads_recv[pid][p][i];
                            }
                        }
                    }
                    
                    std::vector<Ring> z_key_perm(vec_size);
                    std::vector<std::vector<Ring>> z_payloads_perm(num_payloads, std::vector<Ring>(vec_size));
                    std::vector<Ring> z_label_perm(vec_size);
                    
                    for (size_t i = 0; i < vec_size; ++i) {
                        int pi_i = pre_sort->shuffle_pi[preproc_idx][i];
                        z_key_perm[i] = z_key_sum[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                        z_label_perm[i] = z_label_sum[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                        
                        key_shuffled[i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                        label_shuffled[i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                    }
                    for (size_t p = 0; p < num_payloads; ++p) {
                        for (size_t i = 0; i < vec_size; ++i) {
                            int pi_i = pre_sort->shuffle_pi[preproc_idx][i];
                            z_payloads_perm[p][i] = z_payloads_sum[p][pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                            payloads_shuffled[p][i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                        }
                    }
                    
                    network_->send(2, z_key_perm.data(), vec_size * sizeof(Ring));
                    for (size_t p = 0; p < num_payloads; ++p) {
                        network_->send(2, z_payloads_perm[p].data(), vec_size * sizeof(Ring));
                    }
                    network_->send(2, z_label_perm.data(), vec_size * sizeof(Ring));
                    
                } else {
                    network_->send(1, z_key.data(), vec_size * sizeof(Ring));
                    for (size_t p = 0; p < num_payloads; ++p) {
                        network_->send(1, z_payloads[p].data(), vec_size * sizeof(Ring));
                    }
                    network_->send(1, z_label.data(), vec_size * sizeof(Ring));
                    network_->flush(1);
                    
                    std::vector<Ring> z_key_recv(vec_size);
                    std::vector<std::vector<Ring>> z_payloads_recv(num_payloads, std::vector<Ring>(vec_size));
                    std::vector<Ring> z_label_recv(vec_size);
                    
                    network_->recv(id_ - 1, z_key_recv.data(), vec_size * sizeof(Ring));
                    for (size_t p = 0; p < num_payloads; ++p) {
                        network_->recv(id_ - 1, z_payloads_recv[p].data(), vec_size * sizeof(Ring));
                    }
                    network_->recv(id_ - 1, z_label_recv.data(), vec_size * sizeof(Ring));
                    usleep(latency_);
                    
                    std::vector<Ring> z_key_send(vec_size);
                    std::vector<std::vector<Ring>> z_payloads_send(num_payloads, std::vector<Ring>(vec_size));
                    std::vector<Ring> z_label_send(vec_size);
                    
                    for (size_t i = 0; i < vec_size; ++i) {
                        int pi_i = pre_sort->shuffle_pi[preproc_idx][i];
                        
                        if (id_ != nP_) {
                            z_key_send[i] = z_key_recv[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                            z_label_send[i] = z_label_recv[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                            
                            key_shuffled[i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                            label_shuffled[i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                        } else {
                            z_key_send[i] = z_key_recv[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt() - 
                                           pre_sort->shuffle_delta[preproc_idx][i];
                            z_label_send[i] = z_label_recv[pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt() - 
                                             pre_sort->shuffle_delta[preproc_idx][i];
                            
                            key_shuffled[i] = z_key_send[i] + pre_sort->shuffle_c[preproc_idx][i].valueAt();
                            label_shuffled[i] = z_label_send[i] + pre_sort->shuffle_c[preproc_idx][i].valueAt();
                        }
                    }
                    
                    for (size_t p = 0; p < num_payloads; ++p) {
                        for (size_t i = 0; i < vec_size; ++i) {
                            int pi_i = pre_sort->shuffle_pi[preproc_idx][i];
                            
                            if (id_ != nP_) {
                                z_payloads_send[p][i] = z_payloads_recv[p][pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt();
                                payloads_shuffled[p][i] = pre_sort->shuffle_c[preproc_idx][i].valueAt();
                            } else {
                                z_payloads_send[p][i] = z_payloads_recv[p][pi_i] + pre_sort->shuffle_b[preproc_idx][pi_i].valueAt() - 
                                                       pre_sort->shuffle_delta[preproc_idx][i];
                                payloads_shuffled[p][i] = z_payloads_send[p][i] + pre_sort->shuffle_c[preproc_idx][i].valueAt();
                            }
                        }
                    }
                    
                    if (id_ != nP_) {
                        network_->send(id_ + 1, z_key_send.data(), vec_size * sizeof(Ring));
                        for (size_t p = 0; p < num_payloads; ++p) {
                            network_->send(id_ + 1, z_payloads_send[p].data(), vec_size * sizeof(Ring));
                        }
                        network_->send(id_ + 1, z_label_send.data(), vec_size * sizeof(Ring));
                    }
                }
                
                // Step 4: Reconstruct label (don't need to reconstruct key or payloads)
                std::vector<Ring> label_reconstructed(vec_size, 0);
                reconstruct(nP_, id_, network_, label_shuffled, label_reconstructed, use_pking_, latency_);
                
                // Step 5: Apply permutation based on reconstructed labels
                // Reorder key, payloads based on label indices
                std::vector<Ring> key_compacted(vec_size);
                std::vector<std::vector<Ring>> payloads_compacted(num_payloads, std::vector<Ring>(vec_size));
                
                for (size_t i = 0; i < vec_size; ++i) {
                    int idx_perm = static_cast<int>(label_reconstructed[i]);
                    if (idx_perm >= 0 && idx_perm < vec_size) {
                        key_compacted[idx_perm] = key_shuffled[i];
                        for (size_t p = 0; p < num_payloads; ++p) {
                            payloads_compacted[p][idx_perm] = payloads_shuffled[p][i];
                        }
                    }
                }
                
                // Update for next iteration
                bits_current[bit_idx] = key_compacted;
                for (int higher_bit = 31, p = 0; higher_bit > bit_idx; --higher_bit, ++p) {
                    bits_current[higher_bit] = payloads_compacted[p];
                }
                payload_shares = payloads_compacted[num_payloads - 1];
            }
            
            // After all 32 iterations, payload_shares contains the sorting permutation (0-indexed)
            // Now shuffle and reveal the permutation
            std::vector<Ring> z_perm(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                z_perm[i] = payload_shares[i] + pre_sort->final_shuffle_a[i].valueAt();
            }
            
            std::vector<Ring> payload_shuffled(vec_size);
            
            // Final shuffle protocol
            if (id_ == 1) {
                std::vector<std::vector<Ring>> z_perm_recv(nP_);
                z_perm_recv[0] = z_perm;
                
                #pragma omp parallel for
                for (int pid = 2; pid <= nP_; ++pid) {
                    z_perm_recv[pid - 1].resize(vec_size);
                    network_->recv(pid, z_perm_recv[pid - 1].data(), vec_size * sizeof(Ring));
                }
                usleep(latency_);
                
                std::vector<Ring> z_perm_sum(vec_size, 0);
                for (int pid = 0; pid < nP_; ++pid) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_perm_sum[i] += z_perm_recv[pid][i];
                    }
                }
                
                std::vector<Ring> z_perm_perm(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_sort->final_shuffle_pi[i];
                    z_perm_perm[i] = z_perm_sum[pi_i] + pre_sort->final_shuffle_b[pi_i].valueAt();
                    payload_shuffled[i] = pre_sort->final_shuffle_c[i].valueAt();
                }
                
                network_->send(2, z_perm_perm.data(), vec_size * sizeof(Ring));
                
            } else {
                network_->send(1, z_perm.data(), vec_size * sizeof(Ring));
                network_->flush(1);
                
                std::vector<Ring> z_perm_recv(vec_size);
                network_->recv(id_ - 1, z_perm_recv.data(), vec_size * sizeof(Ring));
                usleep(latency_);
                
                std::vector<Ring> z_perm_send(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    int pi_i = pre_sort->final_shuffle_pi[i];
                    
                    if (id_ != nP_) {
                        z_perm_send[i] = z_perm_recv[pi_i] + pre_sort->final_shuffle_b[pi_i].valueAt();
                        payload_shuffled[i] = pre_sort->final_shuffle_c[i].valueAt();
                    } else {
                        z_perm_send[i] = z_perm_recv[pi_i] + pre_sort->final_shuffle_b[pi_i].valueAt() - 
                                        pre_sort->final_shuffle_delta[i];
                        payload_shuffled[i] = z_perm_send[i] + pre_sort->final_shuffle_c[i].valueAt();
                    }
                }
                
                if (id_ != nP_) {
                    network_->send(id_ + 1, z_perm_send.data(), vec_size * sizeof(Ring));
                }
            }
            
            // Reconstruct the permutation
            std::vector<Ring> perm_reconstructed(vec_size, 0);
            reconstruct(nP_, id_, network_, payload_shuffled, perm_reconstructed, use_pking_, latency_);
            
            // Final step: Rearrange all input wires according to revealed permutation
            // For each element, all 32 bits move together
            for (size_t elem = 0; elem < vec_size; ++elem) {
                int perm_idx = static_cast<int>(perm_reconstructed[elem]);
                if (perm_idx >= 0 && perm_idx < vec_size) {
                    // Copy all 32 bits from input element perm_idx to output element elem
                    for (size_t bit = 0; bit < 32; ++bit) {
                        wires_[sort_gate.outs[elem * 32 + bit]] = wires_[sort_gate.in[perm_idx * 32 + bit]];
                    }
                }
            }
        }
    }

    void OnlineEvaluator::deleteWiresEvaluate(const std::vector<common::utils::SIMDOGate> &delete_gates) {
        if (id_ == 0) { return; }
        if (delete_gates.empty()) { return; }

        for (const auto& delete_gate : delete_gates) {
            auto* pre_delete = static_cast<PreprocDeleteWiresGate<Ring>*>(preproc_.gates[delete_gate.out].get());
            
            // Input format: [del_0,...,del_n, p1_0,...,p1_n, p2_0,...,p2_n, ...]
            // Output format: [p1_out_0,...,p1_out_n, p2_out_0,...,p2_out_n, ...] (compacted)
            
            size_t vec_size = delete_gate.vec_size;  // From gate metadata
            size_t total_inputs = delete_gate.in.size();
            size_t num_payloads = (total_inputs / vec_size) - 1;
            
            // Extract del and payload shares
            std::vector<Ring> del_shares(vec_size);
            std::vector<std::vector<Ring>> payload_shares(num_payloads, std::vector<Ring>(vec_size));
            
            for (size_t i = 0; i < vec_size; ++i) {
                del_shares[i] = wires_[delete_gate.in[i]];
            }
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    payload_shares[p][i] = wires_[delete_gate.in[vec_size * (p + 1) + i]];
                }
            }
            
            // Step 1: Shuffle del and all payloads using the same permutation
            std::vector<Ring> shuffled_del(vec_size);
            std::vector<std::vector<Ring>> shuffled_payloads(num_payloads, std::vector<Ring>(vec_size));
            
            // Shuffle logic - mask with 'a' values
            std::vector<Ring> z_del(vec_size);
            std::vector<std::vector<Ring>> z_payloads(num_payloads, std::vector<Ring>(vec_size));
            
            for (size_t i = 0; i < vec_size; ++i) {
                z_del[i] = del_shares[i] + pre_delete->shuffle_a[i].valueAt();
            }
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    z_payloads[p][i] = payload_shares[p][i] + pre_delete->shuffle_a[i].valueAt();
                }
            }
            
            // Shuffle protocol for del and payloads using same permutation
            if (id_ == 1) {
                // Party 1 collects from all parties
                std::vector<Ring> z_del_sum = z_del;
                std::vector<std::vector<Ring>> z_payloads_sum = z_payloads;
                
                for (int pid = 2; pid <= nP_; ++pid) {
                    std::vector<Ring> z_del_recv(vec_size);
                    network_->recv(pid, z_del_recv.data(), vec_size * sizeof(Ring));
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_del_sum[i] += z_del_recv[i];
                    }
                    
                    for (size_t p = 0; p < num_payloads; ++p) {
                        std::vector<Ring> z_payload_recv(vec_size);
                        network_->recv(pid, z_payload_recv.data(), vec_size * sizeof(Ring));
                        for (size_t i = 0; i < vec_size; ++i) {
                            z_payloads_sum[p][i] += z_payload_recv[i];
                        }
                    }
                }
                usleep(latency_);
                
                // Apply permutation and mask for del
                std::vector<Ring> z_del_permuted(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    z_del_permuted[i] = z_del_sum[pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt();
                    shuffled_del[i] = pre_delete->shuffle_c[i].valueAt();
                }
                network_->send(2, z_del_permuted.data(), vec_size * sizeof(Ring));
                
                // Apply permutation and mask for payloads
                for (size_t p = 0; p < num_payloads; ++p) {
                    std::vector<Ring> z_payload_permuted(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        z_payload_permuted[i] = z_payloads_sum[p][pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt();
                        shuffled_payloads[p][i] = pre_delete->shuffle_c[i].valueAt();
                    }
                    network_->send(2, z_payload_permuted.data(), vec_size * sizeof(Ring));
                }
            } else {
                // Send to party 1
                network_->send(1, z_del.data(), vec_size * sizeof(Ring));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->send(1, z_payloads[p].data(), vec_size * sizeof(Ring));
                }
                network_->flush(1);
                
                // Receive from previous party
                std::vector<Ring> z_del_permuted(vec_size);
                network_->recv(id_ - 1, z_del_permuted.data(), vec_size * sizeof(Ring));
                usleep(latency_);
                
                std::vector<std::vector<Ring>> z_payloads_permuted(num_payloads, std::vector<Ring>(vec_size));
                for (size_t p = 0; p < num_payloads; ++p) {
                    network_->recv(id_ - 1, z_payloads_permuted[p].data(), vec_size * sizeof(Ring));
                }
                
                // Apply own permutation for del
                std::vector<Ring> z_del_next(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    if (id_ != nP_) {
                        z_del_next[i] = z_del_permuted[pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt();
                        shuffled_del[i] = pre_delete->shuffle_c[i].valueAt();
                    } else {
                        z_del_next[i] = z_del_permuted[pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt() - pre_delete->shuffle_delta[i];
                        shuffled_del[i] = z_del_next[i] + pre_delete->shuffle_c[i].valueAt();
                    }
                }
                
                if (id_ != nP_) {
                    network_->send(id_ + 1, z_del_next.data(), vec_size * sizeof(Ring));
                }
                
                // Apply own permutation for payloads
                for (size_t p = 0; p < num_payloads; ++p) {
                    std::vector<Ring> z_payload_next(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        if (id_ != nP_) {
                            z_payload_next[i] = z_payloads_permuted[p][pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt();
                            shuffled_payloads[p][i] = pre_delete->shuffle_c[i].valueAt();
                        } else {
                            z_payload_next[i] = z_payloads_permuted[p][pre_delete->shuffle_pi[i]] + pre_delete->shuffle_b[pre_delete->shuffle_pi[i]].valueAt() - pre_delete->shuffle_delta[i];
                            shuffled_payloads[p][i] = z_payload_next[i] + pre_delete->shuffle_c[i].valueAt();
                        }
                    }
                    
                    if (id_ != nP_) {
                        network_->send(id_ + 1, z_payload_next.data(), vec_size * sizeof(Ring));
                    }
                }
            }
            
            // Step 2: Reconstruct del to reveal deletion positions
            std::vector<Ring> del_reconstructed(vec_size, 0);
            reconstruct(nP_, id_, network_, shuffled_del, del_reconstructed, use_pking_, latency_);
            
            // Step 3: Compact payloads by removing indices where del == 1
            // Count non-deleted elements
            std::vector<size_t> keep_indices;
            for (size_t i = 0; i < vec_size; ++i) {
                if (del_reconstructed[i] == 0) {
                    keep_indices.push_back(i);
                }
            }
            
            // Write keep_indices to the first output wire (only party 1 has the actual value)
            // We'll encode the count as the wire value for party 1
            if (id_ == 1) {
                wires_[delete_gate.outs[0]] = static_cast<Ring>(keep_indices.size());
            } else {
                wires_[delete_gate.outs[0]] = Ring(0);
            }
            
            // Write compacted outputs to wires (front-packed)
            // Output format: [keep_indices_wire, p1_0,...,p1_n, p2_0,...,p2_n, ...]
            for (size_t p = 0; p < num_payloads; ++p) {
                for (size_t i = 0; i < vec_size; ++i) {
                    if (i < keep_indices.size()) {
                        wires_[delete_gate.outs[1 + vec_size * p + i]] = shuffled_payloads[p][keep_indices[i]];
                    } else {
                        // Fill remaining slots with zero (or leave as-is)
                        wires_[delete_gate.outs[1 + vec_size * p + i]] = Ring(0);
                    }
                }
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
        std::vector<common::utils::SIMDOGate> compact_gates;
        std::vector<common::utils::SIMDOGate> gi_gates;
        std::vector<common::utils::SIMDOGate> gp_gates;
        std::vector<common::utils::SIMDOGate> sort_gates;
        std::vector<common::utils::SIMDOGate> rewire_gates;
        std::vector<common::utils::SIMDOGate> delete_gates;

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
                case common::utils::GateType::kCompact: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Compact gates are processed immediately since they're complex
                    // compactEvaluate(*g);
                    compact_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kGroupwiseIndex: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Group-wise Index gates are processed immediately
                    // groupwiseIndexEvaluate(*g);
                    gi_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kGroupwisePropagate: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Group-wise Propagate gates are processed immediately
                    groupwisePropagateEvaluate(*g, latency_);
                    // gp_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kSort: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Sort gates are processed immediately
                    sort_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kRewire: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Rewire gates are processed immediately
                    rewire_gates.push_back(*g);
                    break;
                }
                case common::utils::GateType::kDeleteWires: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    // Delete wires gates are processed immediately
                    delete_gates.push_back(*g);
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
        if (!compact_gates.empty()) { compactEvaluateParallel(compact_gates); }
        if (!gi_gates.empty()) { groupwiseIndexEvaluateParallel(gi_gates); }
        // if (!gp_gates.empty()) { groupwisePropagateEvaluateParallel(gp_gates); }
        if (!sort_gates.empty()) { sortEvaluate(sort_gates); }
        if (!rewire_gates.empty()) { rewireEvaluate(rewire_gates); }
        if (!delete_gates.empty()) { deleteWiresEvaluate(delete_gates); }
        
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
                    
                    std::cout << "PublicPerm gate with vec_len: " << vec_len << std::endl;
                    std::cout << "  Output wires: ";
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
                            std::cout << "  Input " << i << " (wire " << g->in[i] << " = " << wires_[g->in[i]] 
                                      << ") -> dynamic idx from wire " << wire_id << " (value=" << wire_value << ", as_int=" << idx_perm 
                                      << ") -> output wire " << g->outs[idx_perm] << std::endl;
                        } else {
                            std::cout << "  Input " << i << " (wire " << g->in[i] << " = " << wires_[g->in[i]] 
                                      << ") -> static idx " << idx_perm 
                                      << " -> output wire " << g->outs[idx_perm] << std::endl;
                        }
                        temp_outputs[g->outs[idx_perm]] = wires_[g->in[i]];
                    }
                    
                    std::cout << "size of temp_outputs: " << temp_outputs.size() << std::endl;
                    // Then, write all outputs at once
                    for (const auto& [wire_id, value] : temp_outputs) {
                        wires_[wire_id] = value;
                        std::cout << "Perm gate at depth " << depth << " output wire " << wire_id << " set to " << value << std::endl;
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
