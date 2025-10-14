#include "online_evaluator.h"

#include "../utils/helpers.h"
#include <omp.h>

namespace graphdb
{
    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit<Ring> preproc,
                                     common::utils::LevelOrderedCircuit circ,
                                     int threads, int seed, int latency)
        : nP_(nP),
          id_(id),
          latency_(latency),
          rgen_(id, seed),
          network_(std::move(network)),
          preproc_(std::move(preproc)),
          circ_(std::move(circ)),
          wires_(circ.num_wires)
    {
        // tpool_ = std::make_shared<ThreadPool>(threads);
    }

    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit<Ring> preproc,
                                     common::utils::LevelOrderedCircuit circ,
                                     std::shared_ptr<ThreadPool> tpool, int seed, int latency)
        : nP_(nP),
          id_(id),
          latency_(latency),
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
        if (id_ != pKing) {
            network_->send(pKing, r1_send.data(), r1_send.size() * sizeof(Ring));
            usleep(latency_);
            network_->recv(pKing, recon_m1.data(), recon_m1.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            // King party adds its own share first
            share_recv[pKing - 1] = r1_send;
            usleep(latency_);
            // Receive from all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    share_recv[pid - 1].resize(num_eqz_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                #pragma omp parallel for
                for (int i = 0; i < num_eqz_gates; ++i) {
                    recon_m1[i] += share_recv[pid][i];
                }
            }
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_m1.data(), recon_m1.size() * sizeof(Ring));
                    network_->flush(pid);
                }
            }
            
        }

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
        if (id_ != pKing) {
            network_->send(pKing, share_m2.data(), share_m2.size() * sizeof(Ring));
            usleep(latency_);
            network_->recv(pKing, recon_m2.data(), recon_m2.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            // King party adds its own share first
            share_recv[pKing - 1] = share_m2;
            usleep(latency_);
            // Receive from all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    share_recv[pid - 1].resize(num_eqz_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                #pragma omp parallel for
                for (int i = 0; i < num_eqz_gates; ++i) {
                    recon_m2[i] += share_recv[pid][i];
                }
            }
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_m2.data(), recon_m2.size() * sizeof(Ring));
                    network_->flush(pid);
                }
            }
        }
 

        // Compute final output share
        std::vector<Ring> recon_out(num_eqz_gates, 0);
        // #pragma omp parallel for
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
        int pKing = 1; // Designated king party
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
        
        auto *pre_rec = static_cast<PreprocRecGate<Ring> *>(preproc_.gates[rec_gates[0].out].get());
        bool via_pking = pre_rec->viaPking;

        if (via_pking) {
            // Reconstruction via king party
            if (id_ != pKing) {
                network_->send(pKing, shares_to_send.data(), shares_to_send.size() * sizeof(Ring));
                usleep(latency_);
                network_->recv(pKing, reconstructed.data(), reconstructed.size() * sizeof(Ring));
            } else {
                std::vector<std::vector<Ring>> share_recv(nP_);
                // King party adds its own share first
                share_recv[pKing - 1] = shares_to_send;
                usleep(latency_);
                // Receive from all other parties
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        share_recv[pid - 1].resize(num_rec_gates);
                        network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                    }
                }
                // Sum all shares
                for (int pid = 0; pid < nP_; ++pid) {
                    #pragma omp parallel for
                    for (size_t i = 0; i < num_rec_gates; ++i) {
                        reconstructed[i] += share_recv[pid][i];
                    }
                }
                // Send reconstructed values back to all parties
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        network_->send(pid, reconstructed.data(), reconstructed.size() * sizeof(Ring));
                        network_->flush(pid);
                    }
                }
            }
        } else {
            // Direct reconstruction (all parties exchange shares)
            std::vector<std::vector<Ring>> share_recv(nP_);
            share_recv[id_ - 1] = shares_to_send;
            
            // Send shares to all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->send(pid, shares_to_send.data(), shares_to_send.size() * sizeof(Ring));
                }
            }
            
            usleep(latency_);
            
            // Receive shares from all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    share_recv[pid - 1].resize(num_rec_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                }
            }
            
            // Sum all shares to reconstruct
            for (int pid = 0; pid < nP_; ++pid) {
                #pragma omp parallel for
                for (size_t i = 0; i < num_rec_gates; ++i) {
                    reconstructed[i] += share_recv[pid][i];
                }
            }
        }

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

            // Aggregate received z values
            for (int pid = 1; pid < nP_; ++pid) {
                size_t idx_vec = 0;
                for (int idx_gate = 0; idx_gate < shuffle_gates.size(); ++idx_gate) {
                    size_t vec_size = shuffle_gates[idx_gate].in.size();
                    std::vector<Ring> z(z_recv_all[pid].begin() + idx_vec, z_recv_all[pid].begin() + idx_vec + vec_size);
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
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (nP_ < 10) { omp_set_num_threads(nP_); }
                else { omp_set_num_threads(10); }
                #pragma omp parallel for
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
            }

            #pragma omp section
            {
                usleep(latency_);
                for (int idx_gate = 0; idx_gate < permAndSh_gates.size(); ++idx_gate) {
                    if (id_ == permAndSh_gates[idx_gate].owner) {
                        auto *pre_permAndSh = static_cast<PreprocPermAndShGate<Ring> *>(preproc_.gates[permAndSh_gates[idx_gate].out].get());
                        size_t vec_size = permAndSh_gates[idx_gate].in.size();
                        std::vector<std::vector<Ring>> z(nP_, std::vector<Ring>(vec_size, 0));
                        #pragma omp parallel for
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
            }
        }
    }

    void OnlineEvaluator::compactEvaluate(const common::utils::SIMDOGate &compact_gate) {
        if (id_ == 0) { return; }
        
        auto *pre_compact = static_cast<PreprocCompactGate<Ring> *>(preproc_.gates[compact_gate.out].get());
        size_t total_size = compact_gate.in.size(); // 2 * vec_size
        size_t vec_size = total_size / 2;
        
        // Extract t and p vectors from inputs
        // Input format: [t0,...,tn, p0,...,pn]
        std::vector<Ring> t_shares(vec_size);
        std::vector<Ring> p_shares(vec_size);
        for (size_t i = 0; i < vec_size; ++i) {
            t_shares[i] = wires_[compact_gate.in[i]];
            p_shares[i] = wires_[compact_gate.in[vec_size + i]];
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
        
        if (pre_compact->viaPking) {
            if (id_ != pKing) {
                network_->send(pKing, shares_to_send.data(), shares_to_send.size() * sizeof(Ring));
                usleep(latency_);
                std::vector<Ring> reconstructed(2 * vec_size);
                network_->recv(pKing, reconstructed.data(), reconstructed.size() * sizeof(Ring));
                for (size_t i = 0; i < vec_size; ++i) {
                    u_reconstructed[i] = reconstructed[2*i];
                    v_reconstructed[i] = reconstructed[2*i + 1];
                }
            } else {
                std::vector<std::vector<Ring>> share_recv(nP_);
                share_recv[pKing - 1] = shares_to_send;
                usleep(latency_);
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        share_recv[pid - 1].resize(2 * vec_size);
                        network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                    }
                }
                for (int pid = 0; pid < nP_; ++pid) {
                    for (size_t i = 0; i < vec_size; ++i) {
                        u_reconstructed[i] += share_recv[pid][2*i];
                        v_reconstructed[i] += share_recv[pid][2*i + 1];
                    }
                }
                std::vector<Ring> reconstructed(2 * vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    reconstructed[2*i] = u_reconstructed[i];
                    reconstructed[2*i + 1] = v_reconstructed[i];
                }
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        network_->send(pid, reconstructed.data(), reconstructed.size() * sizeof(Ring));
                        network_->flush(pid);
                    }
                }
            }
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
        
        // Step 3: Shuffle t, p, and label using the shuffle preprocessing
        // We'll shuffle all three using the same shuffle operation repeated 3 times
        std::vector<Ring> t_shuffled(vec_size);
        std::vector<Ring> p_shuffled(vec_size);
        std::vector<Ring> label_shuffled(vec_size);
        
        // Shuffle t, p, label - simplified version (single shuffle operation)
        // This is a simplified implementation - in practice you'd call shuffleEvaluate 3 times
        // For now, we'll just do the core shuffle logic inline
        
        // TODO: Implement full shuffle for t, p, label
        // For now, just copy the shares (no shuffle for testing)
        for (size_t i = 0; i < vec_size; ++i) {
            t_shuffled[i] = t_shares[i];
            p_shuffled[i] = p_shares[i];
            label_shuffled[i] = label_shares[i];
        }
        
        // Step 4: Reconstruct label_shuffled
        std::vector<Ring> label_reconstructed(vec_size, 0);
        if (id_ != pKing) {
            network_->send(pKing, label_shuffled.data(), label_shuffled.size() * sizeof(Ring));
            usleep(latency_);
            network_->recv(pKing, label_reconstructed.data(), label_reconstructed.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            share_recv[pKing - 1] = label_shuffled;
            usleep(latency_);
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    share_recv[pid - 1].resize(vec_size);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < vec_size; ++i) {
                    label_reconstructed[i] += share_recv[pid][i];
                }
            }
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, label_reconstructed.data(), label_reconstructed.size() * sizeof(Ring));
                    network_->flush(pid);
                }
            }
        }
        
        // Step 5: Apply public permutation based on reconstructed labels
        std::unordered_map<common::utils::wire_t, Ring> temp_t_outputs;
        std::unordered_map<common::utils::wire_t, Ring> temp_p_outputs;
        
        for (size_t i = 0; i < vec_size; ++i) {
            int idx_perm = static_cast<int>(label_reconstructed[i]);
            if (idx_perm >= 0 && idx_perm < vec_size) {
                temp_t_outputs[compact_gate.outs[idx_perm]] = t_shuffled[i];
                temp_p_outputs[compact_gate.outs[vec_size + idx_perm]] = p_shuffled[i];
            }
        }
        
        // Write outputs
        for (const auto& [wire_id, value] : temp_t_outputs) {
            wires_[wire_id] = value;
        }
        for (const auto& [wire_id, value] : temp_p_outputs) {
            wires_[wire_id] = value;
        }
    }

    void OnlineEvaluator::multEvaluate(const std::vector<common::utils::FIn2Gate> &mult_gates) {
        if (id_ == 0) { return; }
        
        size_t num_mult_gates = mult_gates.size();
        if (num_mult_gates == 0) { return; }
        
        int pKing = 1; // Designated king party
        
        // Check if we should use king-based reconstruction (from first mult gate's preproc)
        auto *pre_first = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[mult_gates[0].out].get());
        bool via_pking = pre_first->viaPking;
        
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
        
        if (via_pking) {
            // ============ Reconstruction via King Party ============
            if (id_ != pKing) {
                // Non-king parties: send shares to king, receive reconstructed values
                network_->send(pKing, shares_to_send.data(), shares_to_send.size() * sizeof(Ring));
                usleep(latency_);
                
                std::vector<Ring> reconstructed(2 * num_mult_gates);
                network_->recv(pKing, reconstructed.data(), reconstructed.size() * sizeof(Ring));
                
                // Unpack reconstructed values
                for (size_t i = 0; i < num_mult_gates; ++i) {
                    u_reconstructed[i] = reconstructed[2*i];
                    v_reconstructed[i] = reconstructed[2*i + 1];
                }
            } else {
                // King party: collect all shares, reconstruct, broadcast
                std::vector<std::vector<Ring>> share_recv(nP_);
                share_recv[pKing - 1] = shares_to_send;
                
                usleep(latency_);
                
                // Receive shares from all other parties
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        share_recv[pid - 1].resize(2 * num_mult_gates);
                        network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                    }
                }
                
                // Sum all shares to reconstruct u and v
                for (int pid = 0; pid < nP_; ++pid) {
                    for (size_t i = 0; i < num_mult_gates; ++i) {
                        u_reconstructed[i] += share_recv[pid][2*i];
                        v_reconstructed[i] += share_recv[pid][2*i + 1];
                    }
                }
                
                // Pack reconstructed values for broadcasting
                std::vector<Ring> reconstructed(2 * num_mult_gates);
                for (size_t i = 0; i < num_mult_gates; ++i) {
                    reconstructed[2*i] = u_reconstructed[i];
                    reconstructed[2*i + 1] = v_reconstructed[i];
                }
                
                // Broadcast reconstructed values to all other parties
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        network_->send(pid, reconstructed.data(), reconstructed.size() * sizeof(Ring));
                        network_->flush(pid);
                    }
                }
            }
        } else {
            // ============ Direct Reconstruction (All-to-All) ============
            std::vector<std::vector<Ring>> share_recv(nP_);
            share_recv[id_ - 1] = shares_to_send;
            
            // Send shares to all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    network_->send(pid, shares_to_send.data(), shares_to_send.size() * sizeof(Ring));
                }
            }
            
            usleep(latency_);
            
            // Receive shares from all other parties
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != id_) {
                    share_recv[pid - 1].resize(2 * num_mult_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                }
            }
            
            // Sum all shares to reconstruct u and v
            for (int pid = 0; pid < nP_; ++pid) {
                for (size_t i = 0; i < num_mult_gates; ++i) {
                    u_reconstructed[i] += share_recv[pid][2*i];
                    v_reconstructed[i] += share_recv[pid][2*i + 1];
                }
            }
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

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }

        std::vector<common::utils::FIn2Gate> mult_gates;
        std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::FIn1Gate> rec_gates;
        std::vector<common::utils::SIMDOGate> shuffle_gates;
        std::vector<common::utils::SIMDOGate> permAndSh_gates;

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
                    compactEvaluate(*g);
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
            #pragma omp parallel for
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
