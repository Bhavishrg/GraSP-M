#include "offline_evaluator.h"

#include <NTL/BasicThreadPool.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>

#include "../utils/helpers.h"

namespace graphdb {
OfflineEvaluator::OfflineEvaluator(int nP, int my_id,
                                   std::shared_ptr<io::NetIOMP> network,
                                   common::utils::LevelOrderedCircuit circ,
                                   int threads, int seed, int latency, bool use_pking)
    : nP_(nP),
      id_(my_id),
      latency_(latency),
      use_pking_(use_pking),
      rgen_(my_id, nP, seed), 
      network_(std::move(network)),
      circ_(std::move(circ)) 
      
      {} // tpool_ = std::make_shared<ThreadPool>(threads); }


void OfflineEvaluator::randomShare(int nP, int pid, RandGenPool& rgen, AuthAddShare& share,
                                         std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  std::cout << "Random share for party " << pid << std::endl;
  if (pid == 0) {
    Field val = Field(0);
    Field valn = Field(0);
    Field tag = Field(0);
    Field tagn = Field(0);
    Field key = share.keySh();

    for (int i = 1; i <= nP; i++) {
        rgen.pi(i).random_data(&val, sizeof(Field));
        randomizeZZp(rgen.pi(i), val, sizeof(Field));
        valn += val;
    }
    
    tagn = key * valn;
    share.pushValue(valn);
    share.pushTag(tagn);

    for (int i = 1; i < nP; i++) {
       rgen.pi(i).random_data(&tag, sizeof(Field));
       randomizeZZp(rgen.pi(i), tag, sizeof(Field));
        tagn -= tag;
    }
    rand_sh_sec.push_back(tagn);

  } else {
    Field val;
    Field tag;
    if (pid != nP) {
        rgen.p0().random_data(&val, sizeof(Field));
        randomizeZZp(rgen.p0(), val, sizeof(Field));
        rgen.p0().random_data(&tag, sizeof(Field));
        randomizeZZp(rgen.p0(), tag, sizeof(Field));
        share.pushValue(val);
        share.pushTag(tag);
    } else {
        rgen.p0().random_data(&val, sizeof(Field));
        randomizeZZp(rgen.p0(), val, sizeof(Field));
        share.pushValue(val);
        share.pushTag(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
    }
  }
}

void OfflineEvaluator::randomShareSecret(int nP, int pid, RandGenPool& rgen,
                                         AuthAddShare& share, Field secret,
                                         std::vector<Field>& rand_sh_sec, size_t& idx_rand_sh_sec) {
  if (pid == 0) {

    Field key = share.keySh();

    

    Field val = Field(0);
    Field valn = secret;

    Field tag = key * secret;
    Field tagn = Field(0);

    share.pushValue(valn);
    share.pushTag(tagn);

    

    for (int i = 1; i < nP; i++) {

        randomizeZZp(rgen.pi(i), val, sizeof(Field));
        randomizeZZp(rgen.pi(i), tag, sizeof(Field));
        valn -= val;
        tagn -= tag;
    }
    
    rand_sh_sec.push_back(valn);
    rand_sh_sec.push_back(tagn);

    

  } else {
    Field val;
    Field tag;
    if (pid != nP) {

        randomizeZZp(rgen.p0(), val, sizeof(Field));
        randomizeZZp(rgen.p0(), tag, sizeof(Field));
        share.pushValue(val);
        share.pushTag(tag);

    } else {

        share.pushValue(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
        share.pushTag(rand_sh_sec[idx_rand_sh_sec]);
        idx_rand_sh_sec++;
        
    }
  }
}

void OfflineEvaluator::setWireMasksParty(const std::unordered_map<common::utils::wire_t, int>& input_pid_map, 
                                         std::vector<Field>& rand_sh_sec)  {
  size_t idx_rand_sh_sec = 0;
  size_t b_idx_rand_sh_sec = 0;

  for (const auto& level : circ_.gates_by_level) {
    for (const auto& gate : level) {
      switch (gate->type) {
        // case common::utils::GateType::kInp: {
        //   AuthAddShare<Ring> share_r;
        //   randomShare(nP_, id_, rgen_, share_r, rand_sh_sec, idx_rand_sh_sec);
        //   auto pid = input_pid_map.at(gate->out);
        //   auto pregate = std::make_unique<PreprocInput<Ring>>(pid, share_r);
        //   preproc_.gates[gate->out] = std::move(pregate);
        //   break;
        // }

        case common::utils::GateType::kInp: {
            auto pid = input_pid_map.at(gate->out);

            Field r = Field(0);

            // Sample the mask value
            if (id_ == 0) {
              randomizeZZp(rgen_.pi(pid), r, sizeof(Field));
            }
            else if (pid == id_) {
              randomizeZZp(rgen_.p0(), r, sizeof(Field));
            }

            std::cout << "sampling done " << std::endl;
            
            // Generate authenticated shares of the mask
            AuthAddShare r_sh;
            randomShareSecret(nP_, id_, rgen_, r_sh, r, rand_sh_sec, idx_rand_sh_sec);

            std::cout << "randomShareSecret done " << std::endl;

            auto pregate = std::make_unique<PreprocInput>(pid, r_sh, r);
            preproc_.gates[gate->out] = std::move(pregate);
            break;
        }

        case common::utils::GateType::kRec: {
          auto pregate = std::make_unique<PreprocRecGate>();
          // King party (party 1) receives the reconstructed value
          bool is_king = (id_ == 1);
          pregate->Pking = is_king;
          preproc_.gates[gate->out] = std::move(pregate);
          break;
        }

        case common::utils::GateType::kMul: {
          AuthAddShare triple_a; // Holds shares of a random value a
          AuthAddShare triple_b; // Holds shares of a random value b
          AuthAddShare triple_c; // Holds shares of c=a*b
          randomShare(nP_, id_, rgen_, triple_a, rand_sh_sec, idx_rand_sh_sec);
          randomShare(nP_, id_, rgen_, triple_b, rand_sh_sec, idx_rand_sh_sec);
          Field tp_prod = Field(0);
          if (id_ == 0) { tp_prod = triple_a.valueAt() * triple_b.valueAt(); }
          randomShareSecret(nP_, id_, rgen_, triple_c, tp_prod, rand_sh_sec, idx_rand_sh_sec);
          preproc_.gates[gate->out] =
              std::move(std::make_unique<PreprocMultGate>(triple_a, triple_b, triple_c));
          break;
        }

        default: {
          break;
        }
      }
    }
  }
}


void OfflineEvaluator::setWireMasks(const std::unordered_map<common::utils::wire_t, int>& input_pid_map) {
  std::vector<Field> rand_sh_sec;

  if (id_ == 0) {
    setWireMasksParty(input_pid_map, rand_sh_sec);
    std::cout << "Number of random shares: " << rand_sh_sec.size() << std::endl;

    size_t rand_sh_sec_num = rand_sh_sec.size();
    size_t arith_comm = rand_sh_sec_num;
    std::vector<size_t> lengths(2);
    lengths[0] = arith_comm;
    lengths[1] = rand_sh_sec_num;

    network_->send(nP_, lengths.data(), sizeof(size_t) * lengths.size());

    std::vector<Field> offline_arith_comm(arith_comm);

    for (size_t i = 0; i < rand_sh_sec_num; i++) {
      offline_arith_comm[i] = rand_sh_sec[i];
    }
    network_->send(nP_, offline_arith_comm.data(), sizeof(Field) * arith_comm);

  } else if (id_ != nP_) {
    setWireMasksParty(input_pid_map, rand_sh_sec);

  } else {

    std::vector<size_t> lengths(2);
    usleep(latency_);
    network_->recv(0, lengths.data(), sizeof(size_t) * lengths.size());
    size_t arith_comm = lengths[0];
    size_t rand_sh_sec_num = lengths[1];

    std::vector<Field> offline_arith_comm(arith_comm);
    network_->recv(0, offline_arith_comm.data(), sizeof(Field) * arith_comm);

    rand_sh_sec.resize(rand_sh_sec_num);
    for (int i = 0; i < rand_sh_sec_num; i++) {
      rand_sh_sec[i] = offline_arith_comm[i];
    }

    setWireMasksParty(input_pid_map, rand_sh_sec);
  }
}

PreprocCircuit OfflineEvaluator::getPreproc() {
  return std::move(preproc_);
}

PreprocCircuit OfflineEvaluator::run(const std::unordered_map<common::utils::wire_t, int>& input_pid_map) {
  initializeGlobalKey(nP_, id_, rgen_, network_);

  std::cout << "Running offline evaluation" << std::endl;
  setWireMasks(input_pid_map);

  return std::move(preproc_);
}

};  // namespace graphdb