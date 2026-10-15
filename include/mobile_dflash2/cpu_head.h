// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <mobile_dflash2/cpu_workers.h>
#include <mobile_dflash2/gpu_head.h>
#include <memory>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__ANDROID__)
#include <sched.h>
#include <unistd.h>
#endif
namespace mobile_dflash2 {
class CpuHead {
  uint32_t vocabulary_=0,hidden_=0;
  std::vector<float> scales_;
  std::vector<int8_t> weights_;
  mutable CpuWorkers pool_;
#if defined(__ANDROID__)
  std::unique_ptr<GpuHead> gpu_;
#endif
  bool use_gpu_=false;
  bool verify_gpu_=false;
  mutable bool gpu_verified_=false;
public:
  void use_gpu(bool enabled,bool verify=false){
#if defined(__ANDROID__)
    if(enabled&&!gpu_)gpu_=std::make_unique<GpuHead>(vocabulary_,hidden_,weights_,scales_);
#else
    if(enabled)throw std::runtime_error("GPU draft head is Android-only");
#endif
    use_gpu_=enabled;
    verify_gpu_=verify;gpu_verified_=false;
  }
  bool gpu_verified()const{return gpu_verified_;}
  std::string gpu_name()const {
#if defined(__ANDROID__)
    return use_gpu_?gpu_->device_name():std::string();
#else
    return {};
#endif
  }
  explicit CpuHead(const std::string& file) {
    std::ifstream input(file,std::ios::binary);char magic[8];
    input.read(magic,8);input.read(reinterpret_cast<char*>(&vocabulary_),4);input.read(reinterpret_cast<char*>(&hidden_),4);
    if(!input||std::string(magic,8)!="DF2HEAD1"||!vocabulary_||vocabulary_>262144||!hidden_||hidden_>1536)
      throw std::runtime_error("Invalid packed CPU head contract");
    scales_.resize(vocabulary_);weights_.resize(size_t(vocabulary_)*hidden_);
    input.read(reinterpret_cast<char*>(scales_.data()),scales_.size()*4);
    input.read(reinterpret_cast<char*>(weights_.data()),weights_.size());
    if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Invalid CPU head file size");
    for(float value:scales_)if(!(value>0)||!std::isfinite(value))throw std::runtime_error("Invalid head scale");
  }
  uint32_t vocabulary()const{return vocabulary_;}
  uint32_t hidden_size()const{return hidden_;}
  static int32_t dot(const int8_t* a,const int8_t* b,size_t count) {
    size_t i=0;int32_t total=0;
#if defined(__ARM_NEON)
    int32x4_t sum=vdupq_n_s32(0);
    for(;i+16<=count;i+=16) {
      auto x=vld1q_s8(a+i),y=vld1q_s8(b+i);
#if defined(__ARM_FEATURE_DOTPROD)
      sum=vdotq_s32(sum,x,y);
#else
      sum=vaddq_s32(sum,vpaddlq_s16(vmull_s8(vget_low_s8(x),vget_low_s8(y))));
      sum=vaddq_s32(sum,vpaddlq_s16(vmull_s8(vget_high_s8(x),vget_high_s8(y))));
#endif
    }
    total=vaddvq_s32(sum);
#endif
    for(;i<count;++i)total+=int32_t(a[i])*int32_t(b[i]);
    return total;
  }
  void project(const float* hidden,size_t rows,float* logits,unsigned workers=4,const std::vector<uint32_t>* active=nullptr,bool compact=false)const {
    if(!rows||rows>8||!workers||workers>4)throw std::runtime_error("Invalid CPU head request");
    if(active) {
      if(active->empty())throw std::runtime_error("Empty active vocabulary");
      for(auto token:*active)if(token>=vocabulary_)throw std::runtime_error("Invalid active vocabulary ID");
      if(!compact)std::fill_n(logits,rows*vocabulary_,-INFINITY);
    }
    std::vector<int8_t> queries(rows*hidden_);std::vector<float> query_scales(rows);
    for(size_t r=0;r<rows;++r) {
      float largest=1e-12f;
      for(size_t c=0;c<hidden_;++c) {
        float value=hidden[r*hidden_+c];if(!std::isfinite(value))throw std::runtime_error("Nonfinite draft hidden state");
        largest=std::max(largest,std::abs(value));
      }
      query_scales[r]=largest/127;
      for(size_t c=0;c<hidden_;++c)queries[r*hidden_+c]=int8_t(std::clamp(std::nearbyint(hidden[r*hidden_+c]/query_scales[r]),-127.f,127.f));
    }
    std::vector<float> gpu_reference;
#if defined(__ANDROID__)
    if(use_gpu_) {
      if(!compact)throw std::runtime_error("GPU draft head requires compact logits");
      gpu_->project(queries,query_scales,logits,active);
      if(!verify_gpu_||gpu_verified_)return;
      const size_t count=active?active->size():vocabulary_;
      gpu_reference.assign(logits,logits+rows*count);
    }
#endif
    auto work=[&](unsigned worker) {
      size_t count=active?active->size():size_t(vocabulary_);
      size_t begin=count*worker/workers,end=count*(worker+1)/workers;
      for(size_t index=begin;index<end;++index)for(size_t r=0;r<rows;++r) {
        size_t token=active?(*active)[index]:index;
        const size_t stride=compact&&active?active->size():size_t(vocabulary_);
        logits[r*stride+(compact&&active?index:token)]=float(dot(weights_.data()+token*hidden_,queries.data()+r*hidden_,hidden_))*scales_[token]*query_scales[r];
      }
    };
    pool_.run(workers,work);
    if(!gpu_reference.empty()){
      if(!std::equal(gpu_reference.begin(),gpu_reference.end(),logits))throw std::runtime_error("GPU/CPU INT8 logits differ");
      gpu_verified_=true;
    }
  }
};
}
