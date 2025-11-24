#include "sharing.h"

namespace graphdb {
//check the correctness of the following functions: 
template <>
void AuthAddShare<BoolRing>::randomize(emp::PRG& prg) {
 bool data[3];
 prg.random_bool(static_cast<bool*>(data), 3);
 key_sh_ = data[0];
 value_ = data[1];
 tag_ = data[2];
}

}; // namespace graphdb