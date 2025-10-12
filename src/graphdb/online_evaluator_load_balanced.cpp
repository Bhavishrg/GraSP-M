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
        auto multk_circ = common::utils::Circuit<BoolRing>::generateMultK().orderGatesByLevel();
        size_t num_eqz_gates = eqz_gates.size();
        std::vector<Ring> all_share_send;
        all_share_send.reserve(num_eqz_gates);
        std::vector<preprocg_ptr_t<BoolRing> *> vpreproc;
        vpreproc.reserve(num_eqz_gates);

        // Compute share of d = input + random_value
        for (auto &eqz_gate : eqz_gates) {
            auto *pre_eqz = static_cast<PreprocEqzGate<Ring> *>(preproc_.gates[eqz_gate.out].get());
            Ring share_d = eqz_gate.in + pre_eqz->share_r.valueAt();
            all_share_send.push_back(share_d);
            vpreproc.push_back(pre_eqz->multk_gates.data());
        }

        // Reconstruct the masked input d
        std::vector<Ring> recon_vals(num_eqz_gates, 0);
        if (id_ != pKing) {
            network_->send(pKing, all_share_send.data(), all_share_send.size() * sizeof(Ring));
            usleep(250);
            network_->recv(pKing, recon_vals.data(), recon_vals.size() * sizeof(Ring));
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            usleep(250);
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                share_recv[pid - 1] = std::vector<Ring>();
                if (pid != pKing) {
                    share_recv[pid - 1].resize(num_eqz_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                } else {
                    share_recv[pid - 1].insert(share_recv[pid - 1].begin(), all_share_send.begin(), all_share_send.end());
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                for (int i = 0; i < num_eqz_gates; ++i) {
                    recon_vals[i] += share_recv[pid][i];
                }
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_vals.data(), recon_vals.size() * sizeof(Ring));
                }
            }
        }

        // Evaluate the multK circuit with bits of d as input
        BoolEval bool_eval(id_, nP_, network_, vpreproc, multk_circ);
        for (int i = 0; i < num_eqz_gates; ++i) {
            auto *pre_eqz = static_cast<PreprocEqzGate<Ring> *>(preproc_.gates[eqz_gates[i].out].get());
            Ring recon_d = recon_vals[i];
            auto recon_d_bits = bitDecomposeTwo(recon_d); // Treat as constant
            for (size_t j = 0; j < multk_circ.gates_by_level[0].size(); ++j) {
                const auto &bool_gate = multk_circ.gates_by_level[0][j];
                if (bool_gate->type == common::utils::GateType::kInp) {
                    bool_eval.vwires[i][bool_gate->out] = 1 + recon_d_bits[j] + pre_eqz->share_r_bits[j].valueAt();
                }
            }
        }
        bool_eval.evaluateAllLevels();
        auto output_shares = bool_eval.getOutputShares();

        // Reconstruct the output shares of the multK circuit to compute EQZ and share it in clear
        std::vector<BoolRing> all_out_send(output_shares.size()); // Each EQZ gate has just 1 output wire. So size=1*num_eqz
        for (int i = 0; i < output_shares.size(); ++i) {
            all_out_send[i] = output_shares[i][0];
        }
        std::vector<Ring> recon_out(num_eqz_gates);
        if (id_ != pKing) {
            auto net_data_send = BoolRing::pack(all_out_send.data(), all_out_send.size());
            network_->send(pKing, net_data_send.data(), net_data_send.size() * sizeof(uint8_t));
            usleep(250);
            network_->recv(pKing, recon_out.data(), recon_out.size() * sizeof(Ring));
        } else {
            auto nbytes = (all_out_send.size() + 7) / 8;
            std::vector<std::vector<BoolRing>> all_out_recv(nP_);
            std::vector<BoolRing> out(num_eqz_gates, BoolRing(0));
            usleep(250);
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    std::vector<uint8_t> out_recv(nbytes);
                    network_->recv(pid, out_recv.data(), out_recv.size() * sizeof(uint8_t));
                    all_out_recv[pid - 1] = BoolRing::unpack(out_recv.data(), all_out_send.size());
                } else {
                    all_out_recv[pid - 1] = all_out_send;
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                for (int i = 0; i < num_eqz_gates; ++i) {
                    out[i] += all_out_recv[pid][i];
                }
            }
            for (int i = 0; i < out.size(); ++i) {
                if (out[i] == BoolRing(1)) {
                    recon_out[i] = Ring(1);
                } else {
                    recon_out[i] = Ring(0);
                }
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_out.data(), recon_out.size() * sizeof(Ring));
                }
            }
        }
        for (int i = 0; i < num_eqz_gates; ++i) {
            wires_[eqz_gates[i].out] = recon_out[i]; // Reconstructed output
        }
    }

    void OnlineEvaluator::ltzEvaluate(const std::vector<common::utils::FIn1Gate> &ltz_gates) {
        if (id_ == 0) { return; }
        int pKing = 1; // Designated king party
        auto prefixOR_circ = common::utils::Circuit<BoolRing>::generateParaPrefixOR(2).orderGatesByLevel();
        size_t num_ltz_gates = ltz_gates.size();
        std::vector<Ring> all_share_send;
        all_share_send.reserve(num_ltz_gates);
        std::vector<preprocg_ptr_t<BoolRing> *> vpreproc;
        vpreproc.reserve(num_ltz_gates);

        // Compute share of a = input + random_value
        for (auto &ltz_gate : ltz_gates) {
            auto *pre_ltz = static_cast<PreprocLtzGate<Ring> *>(preproc_.gates[ltz_gate.out].get());
            Ring share_a = ltz_gate.in + pre_ltz->share_r.valueAt();
            all_share_send.push_back(share_a);
            vpreproc.push_back(pre_ltz->PrefixOR_gates.data());
        }

        // Reconstruct the masked input a
        Ring M = pow(2, RINGSIZEBITS - 1); // M = half of ring size
        std::vector<Ring> recon_vals_a(num_ltz_gates, 0); // a = x + r
        std::vector<Ring> recon_vals_b(num_ltz_gates, M); // b = a + M
        if (id_ != pKing) {
            network_->send(pKing, all_share_send.data(), all_share_send.size() * sizeof(Ring));
            usleep(250);
            network_->recv(pKing, recon_vals_a.data(), recon_vals_a.size() * sizeof(Ring));
            for (int i = 0; i < num_ltz_gates; ++i) {
                recon_vals_b[i] += recon_vals_a[i];
            }
        } else {
            std::vector<std::vector<Ring>> share_recv(nP_);
            usleep(250);
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                share_recv[pid - 1] = std::vector<Ring>();
                if (pid != pKing) {
                    share_recv[pid - 1].resize(num_ltz_gates);
                    network_->recv(pid, share_recv[pid - 1].data(), share_recv[pid - 1].size() * sizeof(Ring));
                } else {
                    share_recv[pid - 1].insert(share_recv[pid - 1].begin(), all_share_send.begin(), all_share_send.end());
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                for (int i = 0; i < num_ltz_gates; ++i) {
                    recon_vals_a[i] += share_recv[pid][i];
                    recon_vals_b[i] += share_recv[pid][i];
                }
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_vals_a.data(), recon_vals_a.size() * sizeof(Ring));
                }
            }
        }

        // Evaluate the prefixOR circuit with bits of a and b as inputs
        BoolEval bool_eval(id_, nP_, network_, vpreproc, prefixOR_circ);
        for (int i = 0; i < num_ltz_gates; ++i) {
            auto recon_a_bits = bitDecomposeTwo(recon_vals_a[i]);
            auto recon_b_bits = bitDecomposeTwo(recon_vals_b[i]);
            for (int j = 0; j < prefixOR_circ.gates_by_level[0].size(); ++j) {
                const auto &gate = prefixOR_circ.gates_by_level[0][j];
                if (gate->type == common::utils::GateType::kInp) {
                    if (j < RINGSIZEBITS) {
                        bool_eval.vwires[i][gate->out] = 1 + recon_a_bits[RINGSIZEBITS - 1 - j];
                    } else if (j < 2 * RINGSIZEBITS) {
                        bool_eval.vwires[i][gate->out] = 1 + recon_b_bits[2 * RINGSIZEBITS - 1 - j];
                    } else if (j < 3 * RINGSIZEBITS) {
                        bool_eval.vwires[i][gate->out] = 0;
                    } else {
                        bool_eval.vwires[i][gate->out] = 0;
                    }
                }
            }
        }
        bool_eval.evaluateAllLevels();
        auto output_shares = bool_eval.getOutputShares();

        // Reconstruct the output shares of the prefix_OR circuit to compute LTZ and share it in clear
        std::vector<BoolRing> all_out_send(output_shares.size()); // Each LTZ gate has just 1 output wire. So size=1*num_ltz
        for (int i = 0; i < output_shares.size(); ++i) {
            all_out_send[i] = output_shares[i][0];
        }
        std::vector<Ring> recon_out(num_ltz_gates);
        if (id_ != pKing) {
            auto net_data_send = BoolRing::pack(all_out_send.data(), all_out_send.size());
            network_->send(pKing, net_data_send.data(), net_data_send.size() * sizeof(uint8_t));
            usleep(250);
            network_->recv(pKing, recon_out.data(), recon_out.size() * sizeof(Ring));
        } else {
            auto nbytes = (all_out_send.size() + 7) / 8;
            std::vector<std::vector<BoolRing>> all_out_recv(nP_);
            std::vector<BoolRing> out(num_ltz_gates, BoolRing(0));
            usleep(250);
            #pragma omp parallel for
            for (int pid = 1; pid <= nP_; ++pid) {
                std::vector<uint8_t> out_recv(nbytes);
                if (pid != pKing) {
                    network_->recv(pid, out_recv.data(), out_recv.size() * sizeof(uint8_t));
                    all_out_recv[pid - 1] = BoolRing::unpack(out_recv.data(), all_out_send.size());
                } else {
                    all_out_recv[pid - 1] = all_out_send;
                }
            }
            for (int pid = 0; pid < nP_; ++pid) {
                for (int i = 0; i < num_ltz_gates; ++i) {
                    out[i] += all_out_recv[pid][i];
                }
            }
            for (int i = 0; i < out.size(); ++i) {
                auto lt_bM = recon_vals_b[i] < M; // Locally compute if b<M and XOR to output of prefixOR circuit
                auto ltz = out[i].val() ^ lt_bM;
                if (ltz) {
                    recon_out[i] = Ring(1);
                } else {
                    recon_out[i] = Ring(0);
                }
            }
            for (int pid = 1; pid <= nP_; ++pid) {
                if (pid != pKing) {
                    network_->send(pid, recon_out.data(), recon_out.size() * sizeof(Ring));
                }
            }
        }
        for (int i = 0; i < num_ltz_gates; ++i) {
            wires_[ltz_gates[i].out] = recon_out[i]; // Reconstructed output
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

    void OnlineEvaluator::amortzdPnSEvaluate(const std::vector<common::utils::SIMDMOGate> &amortzdPnS_gates) {
        if (id_ == 0) { return; }
        int pKing = 1; // Designated king party
        std::vector<std::vector<Ring>> z_sum(nP_);
        size_t total_comm = 0;

        for (auto &gate : amortzdPnS_gates) {
            auto *pre_amortzdPnS = static_cast<PreprocAmortzdPnSGate<Ring> *>(preproc_.gates[gate.out].get());
            size_t vec_size = gate.in.size();

            std::vector<Ring> z(vec_size);
            for (int i = 0; i < vec_size; ++i) {
                z[i] = wires_[gate.in[i]] - pre_amortzdPnS->a[i].valueAt();
            }

            std::vector<Ring> z_recon(vec_size, 0);
            if (id_ != pKing) {
                network_->send(pKing, z.data(), z.size() * sizeof(Ring));
                network_->flush(pKing);
                usleep(250);
                network_->recv(pKing, z_recon.data(), z_recon.size() * sizeof(Ring));
            } else {
                usleep(250);
                #pragma omp parallel for
                for (int pid = 1; pid <= nP_; ++pid) {
                    std::vector<Ring> z_recv(vec_size);
                    if (pid != pKing) {
                        network_->recv(pid, z_recv.data(), z_recv.size() * sizeof(Ring));
                        z_sum[pid - 1] = z_recv;
                    } else {
                        z_sum[pid - 1] = z;
                    }
                }
                for (int i = 0; i < vec_size; ++i) {
                    for (int pid = 0; pid < nP_; ++pid) {
                        z_recon[i] += z_sum[pid][i];
                    }
                }
                for (int pid = 1; pid <= nP_; ++pid) {
                    if (pid != pKing) {
                        network_->send(pid, z_recon.data(), z_recon.size() * sizeof(Ring));
                        network_->flush(pid);
                    }
                }
            }

            for (int pid = 0; pid < nP_; ++pid) {
                for (int i = 0; i < vec_size; ++i) {
                    if (pid == id_) {
                        wires_[gate.multi_outs[pid][i]] = z_recon[pre_amortzdPnS->pi[i]] + pre_amortzdPnS->delta[i].valueAt();
                    } else {
                        wires_[gate.multi_outs[pid][i]] = pre_amortzdPnS->b[i].valueAt();
                    }
                }
            }
        }
    }

    void OnlineEvaluator::evaluateGatesAtDepthPartySend(size_t depth, std::vector<Ring> &mult_vals, std::vector<Ring> &mult3_vals,
                                                        std::vector<Ring> &mult4_vals, std::vector<Ring> &dotp_vals) {
        if (id_ == 0) { return; }
        for (auto &gate : circ_.gates_by_level[depth]) {
            switch (gate->type) {
                case common::utils::GateType::kMul: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[g->out].get());
                    auto u = pre_out->triple_a.valueAt() - wires_[g->in1];
                    auto v = pre_out->triple_b.valueAt() - wires_[g->in2];
                    mult_vals.push_back(u);
                    mult_vals.push_back(v);
                    break;
                }

                case common::utils::GateType::kMul3: {
                    auto *g = static_cast<common::utils::FIn3Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMult3Gate<Ring> *>(preproc_.gates[g->out].get());
                    auto u = pre_out->share_a.valueAt() - wires_[g->in1];
                    auto v = pre_out->share_b.valueAt() - wires_[g->in2];
                    auto w = pre_out->share_c.valueAt() - wires_[g->in3];
                    mult3_vals.push_back(u);
                    mult3_vals.push_back(v);
                    mult3_vals.push_back(w);
                    break;
                }

                case common::utils::GateType::kMul4: {
                    auto *g = static_cast<common::utils::FIn4Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMult4Gate<Ring> *>(preproc_.gates[g->out].get());
                    auto u = pre_out->share_a.valueAt() - wires_[g->in1];
                    auto v = pre_out->share_b.valueAt() - wires_[g->in2];
                    auto w = pre_out->share_c.valueAt() - wires_[g->in3];
                    auto x = pre_out->share_d.valueAt() - wires_[g->in4];
                    mult4_vals.push_back(u);
                    mult4_vals.push_back(v);
                    mult4_vals.push_back(w);
                    mult4_vals.push_back(x);
                    break;
                }

                case common::utils::GateType::kDotprod: {
                    auto *g = static_cast<common::utils::SIMDGate *>(gate.get());
                    auto *pre_out = static_cast<PreprocDotpGate<Ring> *>(preproc_.gates[g->out].get());
                    auto vec_len = g->in1.size();
                    for (int i = 0; i < vec_len; ++i) {
                        auto u = pre_out->triple_a_vec[i].valueAt() - wires_[g->in1[i]];
                        auto v = pre_out->triple_b_vec[i].valueAt() - wires_[g->in2[i]];
                        dotp_vals.push_back(u);
                        dotp_vals.push_back(v);
                    }
                    break;
                }

                default:
                    break;
            }
        }
    }

    void OnlineEvaluator::evaluateGatesAtDepthPartyRecv(size_t depth, std::vector<Ring> &mult_vals, std::vector<Ring> &mult3_vals,
                                                        std::vector<Ring> &mult4_vals, std::vector<Ring> &dotp_vals) {
        if (id_ == 0) { return; }
        size_t idx_mult = 0;
        size_t idx_mult3 = 0;
        size_t idx_mult4 = 0;
        size_t idx_dotp = 0;
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

                case common::utils::GateType::kMul: {
                    auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMultGate<Ring> *>(preproc_.gates[g->out].get());
                    Ring u = Ring(0);
                    Ring v = Ring(0);
                    Ring a = pre_out->triple_a.valueAt();
                    Ring b = pre_out->triple_b.valueAt();
                    Ring c = pre_out->triple_c.valueAt();
                    for (int i = 1; i <= nP_; ++i) {
                        u += mult_vals[idx_mult++];
                        v += mult_vals[idx_mult++];
                    }
                    wires_[g->out] = u * v + u * b + v * a + c;
                    break;
                }

                case common::utils::GateType::kMul3: {
                    auto *g = static_cast<common::utils::FIn3Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMult3Gate<Ring> *>(preproc_.gates[g->out].get());
                    Ring u = Ring(0);
                    Ring v = Ring(0);
                    Ring w = Ring(0);
                    Ring a = pre_out->share_a.valueAt();
                    Ring b = pre_out->share_b.valueAt();
                    Ring c = pre_out->share_c.valueAt();
                    Ring ab = pre_out->share_ab.valueAt();
                    Ring bc = pre_out->share_bc.valueAt();
                    Ring ca = pre_out->share_ca.valueAt();
                    Ring abc = pre_out->share_abc.valueAt();
                    for (int i = 1; i <= nP_; ++i) {
                        u += mult3_vals[idx_mult3++];
                        v += mult3_vals[idx_mult3++];
                        w += mult3_vals[idx_mult3++];
                    }
                    wires_[g->out] = (u * v * w) + (u * v * c) + (u * w * b) + (v * w * a) + (u * bc) + (v * ca) + (w * ab) + abc;
                    break;
                }

                case common::utils::GateType::kMul4: {
                    auto *g = static_cast<common::utils::FIn4Gate *>(gate.get());
                    auto *pre_out = static_cast<PreprocMult4Gate<Ring> *>(preproc_.gates[g->out].get());
                    Ring u = Ring(0);
                    Ring v = Ring(0);
                    Ring w = Ring(0);
                    Ring x = Ring(0);
                    Ring a = pre_out->share_a.valueAt();
                    Ring b = pre_out->share_b.valueAt();
                    Ring c = pre_out->share_c.valueAt();
                    Ring d = pre_out->share_c.valueAt();
                    Ring ab = pre_out->share_ab.valueAt();
                    Ring ac = pre_out->share_ac.valueAt();
                    Ring ad = pre_out->share_ad.valueAt();
                    Ring bc = pre_out->share_bc.valueAt();
                    Ring bd = pre_out->share_bd.valueAt();
                    Ring cd = pre_out->share_cd.valueAt();
                    Ring abc = pre_out->share_abc.valueAt();
                    Ring abd = pre_out->share_abd.valueAt();
                    Ring acd = pre_out->share_acd.valueAt();
                    Ring bcd = pre_out->share_bcd.valueAt();
                    Ring abcd = pre_out->share_abcd.valueAt();
                    for (int i = 1; i <= nP_; ++i) {
                        u += mult4_vals[idx_mult4++];
                        v += mult4_vals[idx_mult4++];
                        w += mult4_vals[idx_mult4++];
                        x += mult4_vals[idx_mult4++];
                    }
                    wires_[g->out] = (u * v * w * x) + (u * v * w * d) + (u * v * x * c) + (u * w * x * b) + (v * w * x * a)
                                    + (u * v * cd) + (u * w * bd) + (u * x * bc) + (v * w * ad) + (v * x * ac) + (w * x * ab)
                                    + (u * bcd) + (v * acd) + (w * abd) + (x * abc) + abcd;
                    break;
                }

                case common::utils::GateType::kDotprod: {
                    auto *g = static_cast<common::utils::SIMDGate *>(gate.get());
                    auto *pre_out = static_cast<PreprocDotpGate<Ring> *>(preproc_.gates[g->out].get());
                    auto vec_len = g->in1.size();
                    Ring out = Ring(0);
                    for (int i = 0; i < vec_len; ++i) {
                        Ring u = Ring(0);
                        Ring v = Ring(0);
                        Ring a = pre_out->triple_a_vec[i].valueAt();
                        Ring b = pre_out->triple_b_vec[i].valueAt();
                        Ring c = pre_out->triple_c_vec[i].valueAt();
                        for (int i = 1; i <= nP_; ++i) {
                            u += dotp_vals[idx_dotp++];
                            v += dotp_vals[idx_dotp++];
                        }
                        out += u * v + u * b + v * a + c;
                    }
                    wires_[g->out] = out;
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

                default:
                    break;
            }
        }
    }

    void OnlineEvaluator::evaluateGatesAtDepth(size_t depth) {
        if (id_ == 0) { return; }
        size_t mult_num = 0;
        size_t mult3_num = 0;
        size_t mult4_num = 0;
        size_t dotp_num = 0;
        size_t eqz_num = 0;
        size_t ltz_num = 0;
        size_t shuffle_num = 0;
        size_t permAndSh_num = 0;
        size_t amortzdPnS_num = 0;

        std::vector<Ring> mult_vals;
        std::vector<Ring> mult3_vals;
        std::vector<Ring> mult4_vals;
        std::vector<Ring> dotp_vals;
        std::vector<common::utils::FIn1Gate> eqz_gates;
        std::vector<common::utils::FIn1Gate> ltz_gates;
        std::vector<common::utils::SIMDOGate> shuffle_gates;
        std::vector<common::utils::SIMDOGate> permAndSh_gates;
        std::vector<common::utils::SIMDMOGate> amortzdPnS_gates;

        for (auto &gate : circ_.gates_by_level[depth]) {
            switch (gate->type) {
                case common::utils::GateType::kMul: {
                    mult_num++;
                    break;
                }

                case common::utils::GateType::kMul3: {
                    mult3_num++;
                    break;
                }

                case common::utils::GateType::kMul4: {
                    mult4_num++;
                    break;
                }

                case common::utils::GateType::kDotprod: {
                    dotp_num++;
                    break;
                }

                case ::common::utils::GateType::kEqz: {
                    auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                    eqz_gates.push_back(*g);
                    eqz_num++;
                    break;
                }

                case ::common::utils::GateType::kLtz: {
                    auto *g = static_cast<common::utils::FIn1Gate *>(gate.get());
                    ltz_gates.push_back(*g);
                    ltz_num++;
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

                case common::utils::GateType::kAmortzdPnS: {
                    auto *g = static_cast<common::utils::SIMDMOGate *>(gate.get());
                    amortzdPnS_gates.push_back(*g);
                    amortzdPnS_num++;
                    break;
                }
            }
        }

        if (eqz_num > 0) {
            eqzEvaluate(eqz_gates);
        }

        if (ltz_num > 0) {
            ltzEvaluate(ltz_gates);
        }

        if (shuffle_num > 0) {
            shuffleEvaluate(shuffle_gates);
        }

        if (permAndSh_num > 0) {
            permAndShEvaluate(permAndSh_gates);
        }

        if (amortzdPnS_num > 0) {
            amortzdPnSEvaluate(amortzdPnS_gates);
        }

        evaluateGatesAtDepthPartySend(depth, mult_vals, mult3_vals, mult4_vals, dotp_vals);
        size_t total_comm_send = mult_vals.size() + mult3_vals.size() + mult4_vals.size() + dotp_vals.size();
        size_t total_comm_recv = nP_ * total_comm_send;
        std::vector<Ring> online_comm_send;
        online_comm_send.reserve(total_comm_send);
        std::vector<Ring> online_comm_recv;
        online_comm_recv.reserve(total_comm_recv);
        online_comm_send.insert(online_comm_send.end(), mult_vals.begin(), mult_vals.end());
        online_comm_send.insert(online_comm_send.end(), mult3_vals.begin(), mult3_vals.end());
        online_comm_send.insert(online_comm_send.end(), mult4_vals.begin(), mult4_vals.end());
        online_comm_send.insert(online_comm_send.end(), dotp_vals.begin(), dotp_vals.end());

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

        size_t mult3_all_recv = nP_ * mult3_vals.size();
        std::vector<Ring> mult3_all(mult3_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult3_all_recv;) {
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j];
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j + 1];
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j + 2];
            j += (pid + 1) / nP_;
            pid = (pid + 1) % nP_;
        }

        size_t mult4_all_recv = nP_ * mult4_vals.size();
        std::vector<Ring> mult4_all(mult4_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult4_all_recv;) {
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 1];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 2];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 3];
            j += (pid + 1) / nP_;
            pid = (pid + 1) % nP_;
        }

        size_t dotp_all_recv = nP_ * dotp_vals.size();
        std::vector<Ring> dotp_all(dotp_all_recv);
        for (int i = 0, j = 0, pid = 0; i < dotp_all_recv;) {
            dotp_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + mult4_vals.size() + 2 * j];
            dotp_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + mult4_vals.size() + 2 * j + 1];
            j += (pid + 1) / nP_;
            pid = (pid + 1) % nP_;
        }
        evaluateGatesAtDepthPartyRecv(depth, mult_all, mult3_all, mult4_all, dotp_all);
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

    // Ring OnlineEvaluator::reconstruct(AddShare<Ring> &shares) {
    //     Ring reconstructed_value = Ring(0);
    //     if (id_ != 0) {
    //         for (size_t i = 1; i <= nP_; ++i) {
    //             if (i != id_) {
    //                 network_->send(i, &shares.valueAt(), sizeof(Ring));
    //             }
    //         }
    //         usleep(50000);
    //         for (size_t i = 1; i <= nP_; ++i) {
    //             Ring share_val = Ring(0);
    //             if (i != id_) {
    //                 network_->recv(i, &share_val, sizeof(Ring));
    //             }
    //             reconstructed_value += share_val;
    //         }
    //     }
    //     return reconstructed_value;
    // }

    std::vector<Ring> OnlineEvaluator::evaluateCircuit(const std::unordered_map<common::utils::wire_t, Ring> &inputs) {
        setInputs(inputs);
        for (size_t i = 0; i < circ_.gates_by_level.size(); ++i) {
            evaluateGatesAtDepth(i);
        }
        return getOutputs();
    }

    BoolEval::BoolEval(int my_id, int nP, std::shared_ptr<io::NetIOMP> network,
                       std::vector<preprocg_ptr_t<BoolRing> *> vpreproc,
                       common::utils::LevelOrderedCircuit circ, int seed)
        : id(my_id),
          nP(nP),
          rgen(id, seed),
          network(std::move(network)),
          vwires(vpreproc.size(), std::vector<BoolRing>(circ.num_wires)),
          vpreproc(std::move(vpreproc)),
          circ(std::move(circ)) {}

    void BoolEval::evaluateGatesAtDepthPartySend(size_t depth, std::vector<BoolRing> &mult_vals, std::vector<BoolRing> &mult3_vals,
                                                 std::vector<BoolRing> &mult4_vals, std::vector<BoolRing> &dotp_vals) {
        if (id == 0) { return; }
        for (size_t i = 0; i < vwires.size(); ++i) {
            const auto &preproc = vpreproc[i];
            auto &wires = vwires[i];
            for (auto &gate : circ.gates_by_level[depth]) {
                switch (gate->type) {
                    case common::utils::GateType::kMul: {
                        auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMultGate<BoolRing> *>(preproc[g->out].get());
                        auto u = pre_out->triple_a.valueAt() - wires[g->in1];
                        auto v = pre_out->triple_b.valueAt() - wires[g->in2];
                        mult_vals.push_back(u);
                        mult_vals.push_back(v);
                        break;
                    }

                    case common::utils::GateType::kMul3: {
                        auto *g = static_cast<common::utils::FIn3Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMult3Gate<BoolRing> *>(preproc[g->out].get());
                        auto u = pre_out->share_a.valueAt() - wires[g->in1];
                        auto v = pre_out->share_b.valueAt() - wires[g->in2];
                        auto w = pre_out->share_c.valueAt() - wires[g->in3];
                        mult3_vals.push_back(u);
                        mult3_vals.push_back(v);
                        mult3_vals.push_back(w);
                        break;
                    }

                    case common::utils::GateType::kMul4: {
                        auto *g = static_cast<common::utils::FIn4Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMult4Gate<BoolRing> *>(preproc[g->out].get());
                        auto u = pre_out->share_a.valueAt() - wires[g->in1];
                        auto v = pre_out->share_b.valueAt() - wires[g->in2];
                        auto w = pre_out->share_c.valueAt() - wires[g->in3];
                        auto x = pre_out->share_d.valueAt() - wires[g->in4];
                        mult4_vals.push_back(u);
                        mult4_vals.push_back(v);
                        mult4_vals.push_back(w);
                        mult4_vals.push_back(x);
                        break;
                    }

                    case common::utils::GateType::kDotprod: {
                        auto *g = static_cast<common::utils::SIMDGate *>(gate.get());
                        auto *pre_out = static_cast<PreprocDotpGate<BoolRing> *>(preproc[g->out].get());
                        auto vec_len = g->in1.size();
                        for (int i = 0; i < vec_len; ++i) {
                            auto u = pre_out->triple_a_vec[i].valueAt() - wires[g->in1[i]];
                            auto v = pre_out->triple_b_vec[i].valueAt() - wires[g->in2[i]];
                            dotp_vals.push_back(u);
                            dotp_vals.push_back(v);
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
        }
    }

    void BoolEval::evaluateGatesAtDepthPartyRecv(size_t depth, std::vector<BoolRing> &mult_vals, std::vector<BoolRing> &mult3_vals,
                                                 std::vector<BoolRing> &mult4_vals, std::vector<BoolRing> &dotp_vals) {
        if (id == 0) { return; }
        size_t idx_mult = 0;
        size_t idx_mult3 = 0;
        size_t idx_mult4 = 0;
        size_t idx_dotp = 0;
        for (size_t i = 0; i < vwires.size(); ++i) {
            const auto &preproc = vpreproc[i];
            auto &wires = vwires[i];
            for (auto &gate : circ.gates_by_level[depth]) {
                switch (gate->type) {
                    case common::utils::GateType::kAdd: {
                        auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                        wires[g->out] = wires[g->in1] + wires[g->in2];
                        break;
                    }

                    case common::utils::GateType::kSub: {
                        auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                        wires[g->out] = wires[g->in1] - wires[g->in2];
                        break;
                    }

                    case common::utils::GateType::kConstAdd: {
                        auto *g = static_cast<common::utils::ConstOpGate<BoolRing> *>(gate.get());
                        wires[g->out] = wires[g->in] + g->cval;
                        break;
                    }

                    case common::utils::GateType::kConstMul: {
                        auto *g = static_cast<common::utils::ConstOpGate<BoolRing> *>(gate.get());
                        wires[g->out] = wires[g->in] * g->cval;
                        break;
                    }

                    case common::utils::GateType::kMul: {
                        auto *g = static_cast<common::utils::FIn2Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMultGate<BoolRing> *>(preproc[g->out].get());
                        BoolRing u = 0;
                        BoolRing v = 0;
                        BoolRing a = pre_out->triple_a.valueAt();
                        BoolRing b = pre_out->triple_b.valueAt();
                        BoolRing c = pre_out->triple_c.valueAt();
                        for (int i = 1; i <= nP; ++i) {
                            u += mult_vals[idx_mult++];
                            v += mult_vals[idx_mult++];
                        }
                        wires[g->out] = u * v + u * b + v * a + c;
                        break;
                    }

                    case common::utils::GateType::kMul3: {
                        auto *g = static_cast<common::utils::FIn3Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMult3Gate<BoolRing> *>(preproc[g->out].get());
                        BoolRing u = 0;
                        BoolRing v = 0;
                        BoolRing w = 0;
                        BoolRing a = pre_out->share_a.valueAt();
                        BoolRing b = pre_out->share_b.valueAt();
                        BoolRing c = pre_out->share_c.valueAt();
                        BoolRing ab = pre_out->share_ab.valueAt();
                        BoolRing bc = pre_out->share_bc.valueAt();
                        BoolRing ca = pre_out->share_ca.valueAt();
                        BoolRing abc = pre_out->share_abc.valueAt();
                        for (int i = 1; i <= nP; ++i) {
                            u += mult3_vals[idx_mult3++];
                            v += mult3_vals[idx_mult3++];
                            w += mult3_vals[idx_mult3++];
                        }
                        wires[g->out] = (u * v * w) + (u * v * c) + (u * w * b) + (v * w * a) + (u * bc) + (v * ca) + (w * ab) + abc;
                        break;
                    }

                    case common::utils::GateType::kMul4: {
                        auto *g = static_cast<common::utils::FIn4Gate *>(gate.get());
                        auto *pre_out = static_cast<PreprocMult4Gate<BoolRing> *>(preproc[g->out].get());
                        BoolRing u = 0;
                        BoolRing v = 0;
                        BoolRing w = 0;
                        BoolRing x = 0;
                        BoolRing a = pre_out->share_a.valueAt();
                        BoolRing b = pre_out->share_b.valueAt();
                        BoolRing c = pre_out->share_c.valueAt();
                        BoolRing d = pre_out->share_c.valueAt();
                        BoolRing ab = pre_out->share_ab.valueAt();
                        BoolRing ac = pre_out->share_ac.valueAt();
                        BoolRing ad = pre_out->share_ad.valueAt();
                        BoolRing bc = pre_out->share_bc.valueAt();
                        BoolRing bd = pre_out->share_bd.valueAt();
                        BoolRing cd = pre_out->share_cd.valueAt();
                        BoolRing abc = pre_out->share_abc.valueAt();
                        BoolRing abd = pre_out->share_abd.valueAt();
                        BoolRing acd = pre_out->share_acd.valueAt();
                        BoolRing bcd = pre_out->share_bcd.valueAt();
                        BoolRing abcd = pre_out->share_abcd.valueAt();
                        for (int i = 1; i <= nP; ++i) {
                            u += mult4_vals[idx_mult4++];
                            v += mult4_vals[idx_mult4++];
                            w += mult4_vals[idx_mult4++];
                            x += mult4_vals[idx_mult4++];
                        }
                        wires[g->out] = (u * v * w * x) + (u * v * w * d) + (u * v * x * c) + (u * w * x * b) + (v * w * x * a)
                                        + (u * v * cd) + (u * w * bd) + (u * x * bc) + (v * w * ad) + (v * x * ac) + (w * x * ab)
                                        + (u * bcd) + (v * acd) + (w * abd) + (x * abc) + abcd;
                        break;
                    }

                    case common::utils::GateType::kDotprod: {
                        auto *g = static_cast<common::utils::SIMDGate *>(gate.get());
                        auto *pre_out = static_cast<PreprocDotpGate<BoolRing> *>(preproc[g->out].get());
                        auto vec_len = g->in1.size();
                        BoolRing out = BoolRing(0);
                        for (int i = 0; i < vec_len; ++i) {
                            BoolRing u = BoolRing(0);
                            BoolRing v = BoolRing(0);
                            BoolRing a = pre_out->triple_a_vec[i].valueAt();
                            BoolRing b = pre_out->triple_b_vec[i].valueAt();
                            BoolRing c = pre_out->triple_c_vec[i].valueAt();
                            for (int i = 1; i <= nP; ++i) {
                                u += dotp_vals[idx_dotp++];
                                v += dotp_vals[idx_dotp++];
                            }
                            out += u * v + u * b + v * a + c;
                        }
                        wires[g->out] = out;
                        break;
                    }

                    default:
                        break;
                }
            }
        }
    }

    void BoolEval::evaluateGatesAtDepth(size_t depth) {
        if (id == 0) { return; }
        size_t mult_num = 0;
        size_t mult3_num = 0;
        size_t mult4_num = 0;
        size_t dotp_num = 0;

        for (auto &gate : circ.gates_by_level[depth]) {
            switch (gate->type) {
                case common::utils::GateType::kMul: {
                    mult_num++;
                    break;
                }

                case common::utils::GateType::kMul3: {
                    mult3_num++;
                    break;
                }

                case common::utils::GateType::kMul4: {
                    mult4_num++;
                    break;
                }

                case common::utils::GateType::kDotprod: {
                    dotp_num++;
                    break;
                }
            }
        }

        mult_num *= vwires.size();
        mult3_num *= vwires.size();
        mult4_num *= vwires.size();
        dotp_num *= vwires.size();
        size_t total_comm_send = mult_num + mult3_num + mult4_num + dotp_num;
        size_t total_comm_recv = nP * total_comm_send;
        std::vector<BoolRing> mult_vals;
        std::vector<BoolRing> mult3_vals;
        std::vector<BoolRing> mult4_vals;
        std::vector<BoolRing> dotp_vals;

        evaluateGatesAtDepthPartySend(depth, mult_vals, mult3_vals, mult4_vals, dotp_vals);

        std::vector<BoolRing> online_comm_send;
        online_comm_send.reserve(total_comm_send);
        std::vector<BoolRing> online_comm_recv;
        online_comm_recv.reserve(total_comm_recv);
        online_comm_send.insert(online_comm_send.end(), mult_vals.begin(), mult_vals.end());
        online_comm_send.insert(online_comm_send.end(), mult3_vals.begin(), mult3_vals.end());
        online_comm_send.insert(online_comm_send.end(), mult4_vals.begin(), mult4_vals.end());
        online_comm_send.insert(online_comm_send.end(), dotp_vals.begin(), dotp_vals.end());

        for (int pid = 1; pid <= nP; ++pid) {
            if (pid != id) {
                auto net_data = BoolRing::pack(online_comm_send.data(), total_comm_send);\
                network->send(pid, net_data.data(), sizeof(uint8_t) * net_data.size());\
            }
        }

        usleep(250);
        size_t nbytes = (total_comm_send + 7) / 8;
        std::vector<std::vector<BoolRing>> online_comm_recv_party(nP);
        #pragma omp parallel for
        for (int pid = 1; pid <= nP; ++pid) {
            if (pid != id) {
                std::vector<uint8_t> net_data(nbytes);
                network->recv(pid, net_data.data(), net_data.size());
                online_comm_recv_party[pid - 1] = BoolRing::unpack(net_data.data(), total_comm_send);
            }
        }
        for (int pid = 0; pid < nP; ++pid) {
            if (pid != id - 1) {
                online_comm_recv.insert(online_comm_recv.end(), online_comm_recv_party[pid].begin(), online_comm_recv_party[pid].end());
            } else {
                online_comm_recv.insert(online_comm_recv.end(), online_comm_send.begin(), online_comm_send.end());
            }
        }

        size_t mult_all_recv = nP * mult_vals.size();
        std::vector<BoolRing> mult_all(mult_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult_all_recv;) {
            mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j];
            mult_all[i++] = online_comm_recv[pid * total_comm_send + 2 * j + 1];
            j += (pid + 1) / nP;
            pid = (pid + 1) % nP;
        }

        size_t mult3_all_recv = nP * mult3_vals.size();
        std::vector<BoolRing> mult3_all(mult3_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult3_all_recv;) {
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j];
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j + 1];
            mult3_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + 3 * j + 2];
            j += (pid + 1) / nP;
            pid = (pid + 1) % nP;
        }

        size_t mult4_all_recv = nP * mult4_vals.size();
        std::vector<BoolRing> mult4_all(mult4_all_recv);
        for (int i = 0, j = 0, pid = 0; i < mult4_all_recv;) {
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 1];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 2];
            mult4_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + 4 * j + 3];
            j += (pid + 1) / nP;
            pid = (pid + 1) % nP;
        }

        size_t dotp_all_recv = nP * dotp_vals.size();
        std::vector<BoolRing> dotp_all(dotp_all_recv);
        for (int i = 0, j = 0, pid = 0; i < dotp_all_recv;) {
            dotp_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + mult4_vals.size() + 2 * j];
            dotp_all[i++] = online_comm_recv[pid * total_comm_send + mult_vals.size() + mult3_vals.size() + mult4_vals.size() + 2 * j + 1];
            j += (pid + 1) / nP;
            pid = (pid + 1) % nP;
        }
        evaluateGatesAtDepthPartyRecv(depth, mult_all, mult3_all, mult4_all, dotp_all);
    }

    void BoolEval::evaluateAllLevels() {
        for (size_t i = 0; i < circ.gates_by_level.size(); ++i) {
            evaluateGatesAtDepth(i);
        }
    }

    std::vector<std::vector<BoolRing>> BoolEval::getOutputShares() {
        std::vector<std::vector<BoolRing>> outputs(vwires.size(), std::vector<BoolRing>(circ.outputs.size()));
        for (size_t i = 0; i < vwires.size(); ++i) {
            const auto &wires = vwires[i];
            for (size_t j = 0; j < circ.outputs.size(); ++j) {
                outputs[i][j] = wires[circ.outputs[j]];
            }
        }
        return outputs;
    }
}; // namespace graphdb
