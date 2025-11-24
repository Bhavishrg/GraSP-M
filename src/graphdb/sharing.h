#pragma once

#include <emp-tool/emp-tool.h>

#include <array>
#include <vector>

#include "../utils/helpers.h"
#include "../utils/types.h"
#include "rand_gen_pool.h"

using namespace common::utils;

namespace graphdb {

// Global key share parameter
template <class R>
thread_local R* global_key_sh = nullptr;
thread_local R* global_key_sh_inv = nullptr;

// Forward declarations
class RandGenPool;

namespace io {
class NetIOMP;
}

// Initialize global key shares
// Helper party (pid == 0) samples nP shares using rgen.pi(i) and computes global_key = sum of all shares
// All parties sample their share using rgen.p0()
// 
// For inverse computation:
// - Helper party computes inverse of global_key
// - Helper samples random values for parties 1 to nP-1 using rgen.pi(i)
// - Helper computes last party's share as: inv(global_key) - sum of random shares
// - Helper sends this value to last party (nP)
// - Parties 1 to nP-1 sample their share using rgen.p0()
// - Last party receives its share from helper
//
// Usage:
//   RandGenPool rgen(my_id, num_parties);
//   std::shared_ptr<io::NetIOMP> network = ...;
//   initializeGlobalKey<Ring>(num_parties, my_id, rgen, network);
//   // Now global_key_sh<Ring> and global_key_sh_inv<Ring> are initialized
//
template <class R>
void initializeGlobalKey(int nP, int pid, RandGenPool& rgen, std::shared_ptr<io::NetIOMP> network) {
  if (global_key_sh<R> == nullptr) {
    global_key_sh<R> = new R();
  }
  if (global_key_sh_inv<R> == nullptr) {
    global_key_sh_inv<R> = new R();
  }

  // Initialize global_key_sh
  if (pid == 0) {
    // Helper party samples nP shares and computes the sum
    R global_key = R(0);
    for (int i = 1; i <= nP; i++) {
      R key_share_i;
      rgen.pi(i).random_data(&key_share_i, sizeof(R));
      global_key += key_share_i;
    }
    *global_key_sh<R> = global_key;
    
    // Compute inverse of global key
    R global_key_inv = inv(global_key);
    *global_key_sh_inv<R> = global_key_inv;
    
    // Sample random shares for parties 1 to nP-1
    R sum_inv_shares = R(0);
    for (int i = 1; i < nP; i++) {
      R inv_share_i;
      rgen.pi(i).random_data(&inv_share_i, sizeof(R));
      sum_inv_shares += inv_share_i;
    }
    
    // Compute last party's share
    R last_party_inv_share = global_key_inv - sum_inv_shares;
    
    // Send to last party (nP)
    network->send(nP, &last_party_inv_share, sizeof(R));
    network->flush(nP);
    

  } else if (pid == nP) {
    // Last party receives its inverse share from helper
    network->recv(0, global_key_sh_inv<R>, sizeof(R));
    
    // Sample global_key_sh using p0
    rgen.p0().random_data(global_key_sh<R>, sizeof(R));
  } else {
    // Parties 1 to nP-1 sample their shares using p0
    rgen.p0().random_data(global_key_sh_inv<R>, sizeof(R));
    rgen.p0().random_data(global_key_sh<R>, sizeof(R));
  }
}

// Cleanup function to deallocate global key
template <class R>
void cleanupGlobalKey() {
  if (global_key_sh<R> != nullptr) {
    delete global_key_sh<R>;
    global_key_sh<R> = nullptr;
  }
  if (global_key_sh_inv<R> != nullptr) {
    delete global_key_sh_inv<R>;
    global_key_sh_inv<R> = nullptr;
  }
}

template <class R>
class AuthAddShare {
  // key_sh is now accessed via global parameter
  // value_ will be additive share of my_id and tag_ will be the additive share of of the tag for my_id.
  R value_;
  R tag_;
  
 public:
  AuthAddShare() = default;
  explicit AuthAddShare(R value, R tag)
      : value_{value}, tag_{tag} {}

  void randomize(emp::PRG& prg) {
    randomizeZZp(prg, value_.data(), sizeof(R));
    randomizeZZp(prg, tag_.data(), sizeof(R));
  }

  R& valueAt() { return value_; }
  R& tagAt() { return tag_; }
  R& keySh() { return *global_key_sh<R>; }

  void pushValue(R val) { value_ = val; } 
  void pushTag(R tag) {tag_ = tag; }
  void setKey(R key) { *global_key_sh<R> = key; }
  
  R valueAt() const { return value_; }
  R tagAt() const { return tag_; }
  R keySh() const { return *global_key_sh<R>; }
  //Check this part
  //void randomize(emp::PRG& prg) {
  //  prg.random_data(values_.data(), sizeof(R) * 3); // This step is not clear 
  //}


  // Arithmetic operators.
  AuthAddShare<R>& operator+=(const AuthAddShare<R>& rhs) {
    value_ += rhs.value_;
    tag_ += rhs.tag_;
    // key_sh is now global, no need to update
    return *this;
  }

  // what is "friend"?
  friend AuthAddShare<R> operator+(AuthAddShare<R> lhs,
                                      const AuthAddShare<R>& rhs) {
    lhs += rhs;
    return lhs;
  }

  AuthAddShare<R>& operator-=(const AuthAddShare<R>& rhs) {
    (*this) += (rhs * R(-1));
    return *this;
  }

  friend AuthAddShare<R> operator-(AuthAddShare<R> lhs,
                                      const AuthAddShare<R>& rhs) {
    lhs -= rhs;
    return lhs;
  }

  AuthAddShare<R>& operator*=(const R& rhs) {
    value_ *= rhs;
    tag_ *= rhs;
    return *this;
  }

  friend AuthAddShare<R> operator*(AuthAddShare<R> lhs, const R& rhs) {
    lhs *= rhs;
    return lhs;
  }

  AuthAddShare<R>& operator<<=(const int& rhs) {
    uint64_t value = conv<uint64_t>(value_);
    uint64_t tag = conv<uint64_t>(tag_);
    value <<= rhs;
    tag <<= rhs;
    value_ = value;
    tag_ = tag;
    return *this;
  }

  friend AuthAddShare<R> operator<<(AuthAddShare<R> lhs, const int& rhs) {
    lhs <<= rhs;
    return lhs;
  }

  AuthAddShare<R>& operator>>=(const int& rhs) {
    uint64_t value = conv<uint64_t>(value_);
    uint64_t tag = conv<uint64_t>(tag_);
    value >>= rhs;
    tag >>= rhs;
    value_ = value;
    tag_ = tag;
    return *this;
  }

  friend AuthAddShare<R> operator>>(AuthAddShare<R> lhs, const int& rhs) {
    lhs >>= rhs;
    return lhs;
  }

  AuthAddShare<R>& add(R val, int pid) {
    if (pid == 1) {
      value_ += val;
      tag_ += (*global_key_sh<R>)*val;
    } else {
      tag_ += (*global_key_sh<R>)*val;
    }

    return *this;
  }

  AuthAddShare<R>& addWithAdder(R val, int pid, int adder) {
    if (pid == adder) {
      value_ += val;
      tag_ += (*global_key_sh<R>)*val;
    } else {
      tag_ += (*global_key_sh<R>)*val;
    }

    return *this;
  }

  AuthAddShare<R>& shift() {
    auto bits = bitDecomposeTwo(value_);
    if (bits[63] == 1)
      value_ = 1;
    else
      value_ = 0;
    bits = bitDecomposeTwo(tag_);
    if (bits[63] == 1)
      tag_ = 1;
    else
      tag_ = 0;

    return *this;
  }
  
};

// template <class R>
// class TPShare {
//   std::vector<R> values_;
//   std::vector<R> tags_;

//   public:
//   TPShare() = default;
//   explicit TPShare(std::vector<R> key_sh, std::vector<R> value, std::vector<R> tag)
//       : key_sh_{key_sh}, values_{std::move(value)}, tags_{std::move(tag)} {}

//   // Access share elements.
//   // idx = i retreives value common with party having i.
//   R& operator[](size_t idx) { return values_.at(idx); }
//   // idx = i retreives tag common with party having i.
//   //R& operator()(size_t idx) { return tags_.at(idx); }
  
//   R operator[](size_t idx) const { return values_.at(idx); }
//   //R operator()(size_t idx) { return tags_.at(idx); }

//   R& macKey() { return *global_key_sh<R>; }

//   R& commonValueWithParty(int pid) {
//     return values_.at(pid);
//   }

//   R& commonTagWithParty(int pid) {
//     return tags_.at(pid);
//   }

//   R& commonKeyWithParty(int pid) {
//     return key_sh_.at(pid);
//   }

//   [[nodiscard]] R commonValueWithParty(int pid) const {
//     return values_.at(pid);
//   }

//   [[nodiscard]] R commonTagWithParty(int pid) const {
//     return tags_.at(pid);
//   }

//   [[nodiscard]] R commonKeyWithParty(int pid) const {
//     return key_sh_.at(pid);
//   }

//   void setKey( R key) { *global_key_sh<R> = key; }
//   void pushValues(R val) { values_.push_back(val); }
//   void pushTags(R tag) {tags_.push_back(tag);}
//   void setKeySh( R keysh) {key_sh_.push_back(keysh); }

//   [[nodiscard]] R secret() const { 
//     R res=values_[0];
//     for (int i = 1; i < values_.size(); i++)
//      res+=values_[i];
//     return res;
//   }
//   // Arithmetic operators.
//   TPShare<R>& operator+=(const TPShare<R>& rhs) {
//     for (size_t i = 1; i < values_.size(); i++) {
//       values_[i] += rhs.values_[i];
//       tags_[i] += rhs.tags_[i];
//         key_sh_[i] = rhs.key_sh_[i];
//     }
//     // global_key_sh is now global, no need to update
//     return *this;
//   }

//   friend TPShare<R> operator+(TPShare<R> lhs,
//                                       const TPShare<R>& rhs) {
//     lhs += rhs;
//     return lhs;
//   }

//   TPShare<R>& operator-=(const TPShare<R>& rhs) {
//     (*this) += (rhs * R(-1));
//     return *this;
//   }

//   friend TPShare<R> operator-(TPShare<R> lhs,
//                                       const TPShare<R>& rhs) {
//     lhs -= rhs;
//     return lhs;
//   }

//   TPShare<R>& operator*=(const R& rhs) {
//     for(size_t i = 1; i < values_.size(); i++) {
//       values_[i] *= rhs;
//       tags_[i] *= rhs;
//     }
//     return *this;
//   }

//   friend TPShare<R> operator*(TPShare<R> lhs, const R& rhs) {
//     lhs *= rhs;
//     return lhs;
//   }

//   TPShare<R>& operator<<=(const int& rhs) {
//     for(size_t i = 1; i < values_.size(); i++) {
//         uint64_t value = conv<uint64_t>(values_[i]);
//         uint64_t tag = conv<uint64_t>(tags_[i]);
//         value <<= rhs;
//         tag <<= rhs;
//         values_[i] = value;
//         tags_[i] = tag;
//     }
//     return *this;
//   }

//   friend TPShare<R> operator<<(TPShare<R> lhs, const int& rhs) {
//     lhs <<= rhs;
//     return lhs;
//   }

//   TPShare<R>& operator>>=(const int& rhs) {
//     for(size_t i = 1; i < values_.size(); i++) {
//         uint64_t value = conv<uint64_t>(values_[i]);
//         uint64_t tag = conv<uint64_t>(tags_[i]);
//         value >>= rhs;
//         tag >>= rhs;
//         values_[i] = value;
//         tags_[i] = tag;
//     }
//     return *this;
//   }

//   friend TPShare<R> operator>>(TPShare<R> lhs, const int& rhs) {
//     lhs >>= rhs;
//     return lhs;
//   }

//   AuthAddShare<R> getAAS(size_t pid){
//     return AuthAddShare<R>({key_sh_.at(pid), values_.at(pid), tags_.at(pid)});
//   }

//   TPShare<R>& shift() {
//     for(size_t i = 1; i < values_.size(); i++) {
//       auto bits = bitDecomposeTwo(values_[i]);
//       if(bits[63] == 1)
//         values_[i] = 1;
//       else 
//         values_[i] = 0;

//       bits = bitDecomposeTwo(tags_[i]);
//       if(bits[63] == 1)
//         tags_[i] = 1;
//       else 
//         tags_[i] = 0;
//     }
//     return *this;
//   }

//   //Add above
  
// };  

};  // namespace graphdb