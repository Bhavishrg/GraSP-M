#include "online_evaluator.h"

#include "../utils/helpers.h"

namespace graphdb
{
    // Helper function to reconstruct shares towards a designated party
    void OnlineEvaluator::reconstructToParty(int nP, int pid, std::shared_ptr<io::NetIOMP> network,
                                            const std::vector<Ring>& shares_list,
                                            std::vector<Ring>& reconstructed_list,
                                            int target_party, int latency) {
        size_t num_shares = shares_list.size();
        reconstructed_list.resize(num_shares, 0);
        
        if (pid == target_party) {
            // Target party receives shares from all other parties
            std::vector<std::vector<Ring>> share_recv(nP);
            share_recv[target_party - 1] = shares_list;
            
            usleep(latency);
            
            // Receive from all parties except self
            for (int p = 1; p <= nP; ++p) {
                if (p != target_party) {
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
        } else {
            // Non-target parties send their shares to target party
            network->send(target_party, shares_list.data(), shares_list.size() * sizeof(Ring));
            network->flush(target_party);
        }
    }

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
                                network_->send(pid, output_shares[id_ - 1].data(), output_shares[id_ - 1].size() * sizeof(Ring));
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

        if (!mult_gates.empty()) {  }
        if (!eqz_gates.empty()) {  }
        if (!rec_gates.empty()) { recEvaluate(rec_gates); }
        if (!shuffle_gates.empty()) {  }
        if (!permAndSh_gates.empty()) {  }
        if (!amortzdPnS_gates.empty()) {  }
        if (!rewire_gates.empty()) {  }
        
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