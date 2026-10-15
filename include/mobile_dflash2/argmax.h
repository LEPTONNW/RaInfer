// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace mobile_dflash2 {
// Keep std::max_element's first-index tie rule for quantized logits.
inline size_t argmax_u16(const uint16_t* values,size_t size) {
  if(!size)throw std::invalid_argument("Empty logits");
  size_t i=0;uint16_t best=0;
#if defined(__ARM_NEON)
  uint16x8_t maximum=vdupq_n_u16(0);
  for(;i+8<=size;i+=8)maximum=vmaxq_u16(maximum,vld1q_u16(values+i));
  best=vmaxvq_u16(maximum);
#endif
  for(;i<size;++i)best=std::max(best,values[i]);
  i=0;
#if defined(__ARM_NEON)
  auto target=vdupq_n_u16(best);
  for(;i+8<=size;i+=8)if(vmaxvq_u16(vceqq_u16(vld1q_u16(values+i),target)))break;
#endif
  for(;i<size;++i)if(values[i]==best)return i;
  throw std::runtime_error("Logit maximum missing");
}
}
