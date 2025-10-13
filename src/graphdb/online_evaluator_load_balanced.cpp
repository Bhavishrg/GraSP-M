#include "online_evaluator.h"

#include "../utils/helpers.h"
#include <omp.h>

namespace graphdb
{
    OnlineEvaluator::OnlineEvaluator(int nP, int id, std::shared_ptr<io::NetIOMP> network,
                                     PreprocCircuit<Ring> preproc,
                                     common::utils::LevelOrderedCircuit circ,
                                     int threads, int seed)
        : nP_(nP),
          id_(id),
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
                                     std::shared_ptr<ThreadPool> tpool, int seed)
        : nP_(nP),
          id_(id),
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
            usleep(250);
            network_->recv(pKing, recon_m1.data(), recon_m1.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            // King party adds its own share first
            share_recv[pKing - 1] = r1_send;
            usleep(250);
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
            usleep(250);
            network_->recv(pKing, recon_m2.data(), recon_m2.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            // King party adds its own share first
            share_recv[pKing - 1] = share_m2;
            usleep(250);
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


    void OnlineEvaluator::shuffleEvaluate(const std::vector<common::utils::SIMDOGate> &shuffle_gates) {
        if (id_ == 0) { return; }
        std::vector<Ring> z_all;
        std::vector<std::vector<Ring>> z_sum;
        size_t total_comm = 0;

        for (auto &gate : shuffle_gates) {
            auto *pre_shuffle = static_cast<PreprocShuffleGate<Ring> *>(preproc_.gates[gate.out].get());
            size_t vec_size = gate.in.size();
            total_comm += vec_size;
            std::vector<Ring> z(vec_size, 0);
            if (id_ != 1) {
                for (int i = 0; i < vec_size; ++i) {
                    z[i] = wires_[gate.in[i]] - pre_shuffle->a[i].valueAt();
                }
                z_all.insert(z_all.end(), z.begin(), z.end());
            } else {
                z_sum.push_back(z);
            }
        }

        if (id_ == 1) {
            usleep(250);
            std::vector<std::vector<Ring>> z_recv_all(nP_);
            #pragma omp parallel for
            for (int pid = 2; pid <= nP_; ++pid) {
                z_recv_all[pid - 1] = std::vector<Ring>(total_comm);
                network_->recv(pid, z_recv_all[pid - 1].data(), z_recv_all[pid - 1].size() * sizeof(Ring));
            }
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
                for (int i = 0; i < vec_size; ++i) {
                    z[i] = z_sum[idx_gate][pre_shuffle->pi[i]] + wires_[shuffle_gates[idx_gate].in[pre_shuffle->pi[i]]]
                           - pre_shuffle->c[i].valueAt();
                    wires_[shuffle_gates[idx_gate].outs[i]] = pre_shuffle->b[i].valueAt();
                }
                z_all.insert(z_all.end(), z.begin(), z.end());
            }
            network_->send(2, z_all.data(), z_all.size() * sizeof(Ring));
        } else {
            network_->send(1, z_all.data(), z_all.size() * sizeof(Ring));
            network_->flush(1);

            z_all.clear();
            z_all.resize(total_comm);
            network_->recv(id_ - 1, z_all.data(), z_all.size() * sizeof(Ring));
            usleep(250);
            for (int idx_gate = 0, idx_vec = 0; idx_gate < shuffle_gates.size(); ++idx_gate) {
                auto *pre_shuffle = static_cast<PreprocShuffleGate<Ring> *>(preproc_.gates[shuffle_gates[idx_gate].out].get());
                size_t vec_size = shuffle_gates[idx_gate].in.size();
                std::vector<Ring> z(z_all.begin() + idx_vec, z_all.begin() + idx_vec + vec_size);
                std::vector<Ring> z_send(vec_size);
                for (int i = 0; i < vec_size; ++i) {
                    if (id_ != nP_) {
                        z_send[i] = z[pre_shuffle->pi[i]] - pre_shuffle->c[i].valueAt();
                        wires_[shuffle_gates[idx_gate].outs[i]] = pre_shuffle->b[i].valueAt();
                    } else {
                        z_send[i] = z[pre_shuffle->pi[i]] + pre_shuffle->delta[i].valueAt();
                        wires_[shuffle_gates[idx_gate].outs[i]] = z_send[i];
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
                usleep(250);
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

    void OnlineEvaluator::multEvaluate(const std::vector<common::utils::FIn2Gate> &mult_gates) {

        std::vector<Ring> mult_vals;
        if (id_ == 0) { return; }
        size_t idx_mult = 0;
        for (auto &mult_gate : mult_gates) {
            auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[mult_gate.out].get());
            auto u = pre_out->triple_a.valueAt() - wires_[mult_gate.in1];
            auto v = pre_out->triple_b.valueAt() - wires_[mult_gate.in2];
            mult_vals.push_back(u);
            mult_vals.push_back(v);
            break;
        }

        size_t total_comm_send = mult_vals.size();
        size_t total_comm_recv = nP_ * total_comm_send;
        std::vector<Ring> online_comm_send;
        online_comm_send.reserve(total_comm_send);
        std::vector<Ring> online_comm_recv;
        online_comm_recv.reserve(total_comm_recv);
        online_comm_send.insert(online_comm_send.end(), mult_vals.begin(), mult_vals.end());

        for (int pid = 1; pid <= nP_; ++pid) {
            if (pid != id_) {
                network_->send(pid, online_comm_send.data(), sizeof(Ring) * online_comm_send.size());
            }
        }

        usleep(250);
        std::vector<std::vector<Ring>> online_comm_recv_party(nP_);
        #pragma omp parallel for
        for (int pid = 1; pid <= nP_; ++pid) {
            if (pid != id_) {
                online_comm_recv_party[pid - 1] = std::vector<Ring>(total_comm_send);
                network_->recv(pid, online_comm_recv_party[pid - 1].data(), sizeof(Ring) * online_comm_recv_party[pid - 1].size());
            }
        }
        for (int pid = 0; pid < nP_; ++pid) {
            if (pid != id_ - 1) {
                online_comm_recv.insert(online_comm_recv.end(), online_comm_recv_party[pid].begin(), online_comm_recv_party[pid].end());
            } else {
                online_comm_recv.insert(online_comm_recv.end(), online_comm_send.begin(), online_comm_send.end());
            }
        }

        size_t mult_all_recv = nP_ * mult_vals.size();
        std::vector<Ring> mult_all(mult_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult_all_recv;) {
            mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j];
            mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j + 1];
            j += (pid + 1) / nP_;
            pid = (pid + 1) % nP_;
        }

        for (auto &mult_gate : mult_gates) {
            auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[mult_gate.out].get());
            Ring u = Ring(0);
            Ring v = Ring(0);
            Ring a = pre_out->triple_a.valueAt();
            Ring b = pre_out->triple_b.valueAt();
            Ring c = pre_out->triple_c.valueAt();
            for (int i = 1; i <= nP_; ++i) {
                u += mult_vals[idx_mult++];
                v += mult_vals[idx_mult++];
            }
            wires_[mult_gate.out] = u * v + u * b + v * a + c;
            break;
        }


    }

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }
        size_t mult_num = 0;
        size_t dotp_num = 0;
        size_t eqz_num = 0;
        size_t shuffle_num = 0;
        size_t permAndSh_num = 0;

        
        std::vector<common::utils::FIn2Gate> mult_gates;
        std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::SIMDOGate> shuffle_gates;
        std::vector<common::utils::SIMDOGate> permAndSh_gates;

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
                    if (id_ == 1) { wires_[g->out] = wires_[g->in] + g->cval; } // Only 1 party needs to add the constant
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
                        auto idx_perm = g->permutation[0][i];
                        wires_[g->outs[idx_perm]] = wires_[g->in[i]];
                    }
                    break;
                }

                case common::utils::GateType::kMul: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    mult_gates.push_back(*g);
                    mult_num++;
                    break;
                }

                case ::common::utils::GateType::kEqz: {
                    auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                    eqz_gates.push_back(*g);
                    eqz_num++;
                    break;
                }


                case common::utils::GateType::kShuffle: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    shuffle_gates.push_back(*g);
                    shuffle_num++;
                    break;
                }

                case common::utils::GateType::kPermAndSh: {
                    auto *g = static_cast<common::utils::SIMDOGate *>(gate.get());
                    permAndSh_gates.push_back(*g);
                    permAndSh_num++;
                    break;
                }

                default:
                    break;
            }
        }



        if (mult_num > 0) {
            multEvaluate(mult_gates);
        }

        if (eqz_num > 0) {
            eqzEvaluate(eqz_gates);
        }

        if (shuffle_num > 0) {
            shuffleEvaluate(shuffle_gates);
        }

        if (permAndSh_num > 0) {
            permAndShEvaluate(permAndSh_gates);
        }

        // evaluateGatesAtDepthPartySend(depth, mult_vals);
        // size_t total_comm_send = mult_vals.size();
        // size_t total_comm_recv = nP_ * total_comm_send;
        // std::vector<Ring> online_comm_send;
        // online_comm_send.reserve(total_comm_send);
        // std::vector<Ring> online_comm_recv;
        // online_comm_recv.reserve(total_comm_recv);
        // online_comm_send.insert(online_comm_send.end(), mult_vals.begin(), mult_vals.end());

        // for (int pid = 1; pid <= nP_; ++pid) {
        //     if (pid != id_) {
        //         network_->send(pid, online_comm_send.data(), sizeof(Ring) * online_comm_send.size());
        //     }
        // }

        // usleep(250);
        // std::vector<std::vector<Ring>> online_comm_recv_party(nP_);
        // #pragma omp parallel for
        // for (int pid = 1; pid <= nP_; ++pid) {
        //     if (pid != id_) {
        //         online_comm_recv_party[pid - 1] = std::vector<Ring>(total_comm_send);
        //         network_->recv(pid, online_comm_recv_party[pid - 1].data(), sizeof(Ring) * online_comm_recv_party[pid - 1].size());
        //     }
        // }
        // for (int pid = 0; pid < nP_; ++pid) {
        //     if (pid != id_ - 1) {
        //         online_comm_recv.insert(online_comm_recv.end(), online_comm_recv_party[pid].begin(), online_comm_recv_party[pid].end());
        //     } else {
        //         online_comm_recv.insert(online_comm_recv.end(), online_comm_send.begin(), online_comm_send.end());
        //     }
        // }

        // size_t mult_all_recv = nP_ * mult_vals.size();
        // std::vector<Ring> mult_all(mult_all_recv);
        // for (int i = 0, j = 0, pid = 0; i < mult_all_recv;) {
        //     mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j];
        //     mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j + 1];
        //     j += (pid + 1) / nP_;
        //     pid = (pid + 1) % nP_;
        // }

        // evaluateGatesAtDepthPartyRecv(depth, mult_all);
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
            usleep(250);
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
