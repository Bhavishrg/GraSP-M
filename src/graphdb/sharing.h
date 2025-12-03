#pragma once

#include <emp-tool/emp-tool.h>

#include <array>
#include <vector>

#include "../utils/helpers.h"
#include "../utils/types.h"
#include "rand_gen_pool.h"
#include "../io/netmp.h"

using namespace common::utils;

namespace graphdb {

extern Field* global_key_sh;
extern Field* global_key_sh_inv;

class RandGenPool;

using NTL::inv;

// Initialize global key shares for Field type
inline void initializeGlobalKey(int nP, int pid, RandGenPool& rgen, std::shared_ptr<::io::NetIOMP> network) {

  std::cout << "Initializing global key for type: " << typeid(Field).name() << std::endl;
  
  network->sync();

  if (global_key_sh == nullptr) {
    global_key_sh = new Field();
  }
  if (global_key_sh_inv == nullptr) {
    global_key_sh_inv = new Field();
  }

  // Initialize global_key_sh
  if (pid == 0) {
    // Helper party samples nP shares and computes the sum
    Field global_key;
    NTL::clear(global_key);
    
    // Keep sampling until we get a non-zero global key
    do {
      NTL::clear(global_key);
      for (int i = 1; i <= nP; i++) {
        Field key_share_i;
        randomizeZZp(rgen.pi(i), key_share_i, sizeof(Field));
        global_key += key_share_i;
      }
    } while (IsZero(global_key));
    
    *global_key_sh = global_key;

    // Compute inverse of global key
    Field global_key_inv = inv(global_key);
    *global_key_sh_inv = global_key_inv;
    
    // Sample random shares for parties 1 to nP-1
    Field sum_inv_shares;
    NTL::clear(sum_inv_shares);
    for (int i = 1; i < nP; i++) {
      Field inv_share_i;
      randomizeZZp(rgen.pi(i), inv_share_i, sizeof(Field));
      sum_inv_shares += inv_share_i;
    }
    
    // Compute last party's share
    Field last_party_inv_share = global_key_inv - sum_inv_shares;

    
    // Send to last party (nP)
    network->send(nP, &last_party_inv_share, sizeof(Field));
    network->flush(nP);

    std::cout << "Global key initialized by party 0: " << *global_key_sh << std::endl;
    std::cout << "Global key inverse initialized by party 0: " << *global_key_sh_inv << std::endl;

  } else if (pid == nP) {
    // Sample global_key_sh using p0
    randomizeZZp(rgen.p0(), *global_key_sh, sizeof(Field));

    // Last party receives its inverse share from helper
    // Small sync to make sure send is ready
    network->recv(0, global_key_sh_inv, sizeof(Field));

    std::cout << "Global key initialized by party 0: " << *global_key_sh << std::endl;
    std::cout << "Global key inverse initialized by party 0: " << *global_key_sh_inv << std::endl;
  } else {
    // Parties 1 to nP-1 sample their shares using p0
    randomizeZZp(rgen.p0(), *global_key_sh, sizeof(Field));
    randomizeZZp(rgen.p0(), *global_key_sh_inv, sizeof(Field));

    std::cout << "Global key initialized by party 0: " << *global_key_sh << std::endl;
    std::cout << "Global key inverse initialized by party 0: " << *global_key_sh_inv << std::endl;
  }

  // Final sync to ensure all parties completed init
  network->sync(); 
}

// Cleanup function to deallocate global key
inline void cleanupGlobalKey() {
  if (global_key_sh != nullptr) {
    delete global_key_sh;
    global_key_sh = nullptr;
  }
  if (global_key_sh_inv != nullptr) {
    delete global_key_sh_inv;
    global_key_sh_inv = nullptr;
  }
}

class AuthAddShare {
  // Field type only - value and tag are additive shares
  Field value_;
  Field tag_;
  
 public:
  AuthAddShare() = default;
  explicit AuthAddShare(Field value, Field tag)
      : value_{value}, tag_{tag} {}

  void randomize(emp::PRG& prg) {
    randomizeZZp(prg, value_, sizeof(Field));
    randomizeZZp(prg, tag_, sizeof(Field));
  }

  Field& valueAt() { return value_; }
  Field& tagAt() { return tag_; }
  Field keySh() const { 
    return *global_key_sh;
  }

  void pushValue(Field val) { value_ = val; } 
  void pushTag(Field tag) {tag_ = tag; }
  void setKey(Field key) { *global_key_sh = key; }


  // Arithmetic operators.
  AuthAddShare& operator+=(const AuthAddShare& rhs) {
    value_ += rhs.value_;
    tag_ += rhs.tag_;
    return *this;
  }

  friend AuthAddShare operator+(AuthAddShare lhs, const AuthAddShare& rhs) {
    lhs += rhs;
    return lhs;
  }

  AuthAddShare& operator-=(const AuthAddShare& rhs) {
    (*this) += (rhs * Field(-1));
    return *this;
  }

  friend AuthAddShare operator-(AuthAddShare lhs, const AuthAddShare& rhs) {
    lhs -= rhs;
    return lhs;
  }

  AuthAddShare& operator*=(const Field& rhs) {
    value_ *= rhs;
    tag_ *= rhs;
    return *this;
  }

  friend AuthAddShare operator*(AuthAddShare lhs, const Field& rhs) {
    lhs *= rhs;
    return lhs;
  }

  AuthAddShare& operator<<=(const int& rhs) {
    uint64_t value = conv<uint64_t>(value_);
    uint64_t tag = conv<uint64_t>(tag_);
    value <<= rhs;
    tag <<= rhs;
    value_ = value;
    tag_ = tag;
    return *this;
  }

  friend AuthAddShare operator<<(AuthAddShare lhs, const int& rhs) {
    lhs <<= rhs;
    return lhs;
  }

  AuthAddShare& operator>>=(const int& rhs) {
    uint64_t value = conv<uint64_t>(value_);
    uint64_t tag = conv<uint64_t>(tag_);
    value >>= rhs;
    tag >>= rhs;
    value_ = value;
    tag_ = tag;
    return *this;
  }

  friend AuthAddShare operator>>(AuthAddShare lhs, const int& rhs) {
    lhs >>= rhs;
    return lhs;
  }

  AuthAddShare& add(Field val, int pid) {
    if (pid == 1) {
      value_ += val;
      tag_ += (*global_key_sh)*val;
    } else {
      tag_ += (*global_key_sh)*val;
    }
    return *this;
  }

  AuthAddShare& addWithAdder(Field val, int pid, int adder) {
    if (pid == adder) {
      value_ += val;
      tag_ += (*global_key_sh)*val;
    } else {
      tag_ += (*global_key_sh)*val;
    }
    return *this;
  }

  AuthAddShare& shift() {
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


};  // namespace graphdb