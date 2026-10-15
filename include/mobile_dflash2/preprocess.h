// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace mobile_dflash2 {
inline void prepare_external(const int64_t* contexts,const bool* valid,const int64_t* queries,size_t width,double theta,
    float* qcos,float* qsin,float* kcos,float* ksin,float* mask) {
  if(!width||!(theta>0)||!std::isfinite(theta))throw std::invalid_argument("Invalid compact preprocessing contract");
  for(size_t row=0;row<width;++row)if(queries[row]<0||queries[row]!=queries[0]+static_cast<int64_t>(row))
    throw std::invalid_argument("Invalid pending block positions");
  for(size_t col=0;col<512;++col)if(valid[col]&&(contexts[col]<0||contexts[col]>=queries[0]))
    throw std::invalid_argument("Future teacher context is not allowed");
  auto rotary=[&](int64_t p,float* cos,float* sin) {
    for(size_t i=0;i<32;++i) {
      float frequency=std::pow(static_cast<float>(theta),-static_cast<float>(2*i)/64.0f);
      float angle=static_cast<float>(p)*frequency;
      cos[i]=cos[i+32]=std::cos(angle);sin[i]=sin[i+32]=std::sin(angle);
    }
  };
  for(size_t row=0;row<width;++row)rotary(queries[row],qcos+row*64,qsin+row*64);
  for(size_t col=0;col<512+width;++col) {
    auto p=col<512?contexts[col]:queries[col-512];rotary(p,kcos+col*64,ksin+col*64);
    for(size_t row=0;row<width;++row)mask[row*(512+width)+col]=((col>=512||valid[col])&&p>=queries[row]-511)?0.0f:-65535.0f;
  }
}
}
