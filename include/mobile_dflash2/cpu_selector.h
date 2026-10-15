// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <mobile_dflash2/cpu_head.h>
namespace mobile_dflash2 {
class CpuSelector {
  uint32_t vocabulary_=0,hidden_=0,rank_=0,topk_=0;
  std::vector<float> projection_,predecessors_,successors_;
  static float dot(const float*a,const float*b,size_t n){float result=0;for(size_t i=0;i<n;++i)result+=a[i]*b[i];return result;}
public:
  explicit CpuSelector(const std::string&path){
    std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);
    f.read(reinterpret_cast<char*>(&vocabulary_),4);f.read(reinterpret_cast<char*>(&hidden_),4);f.read(reinterpret_cast<char*>(&rank_),4);f.read(reinterpret_cast<char*>(&topk_),4);
    if(!f||std::string(magic,8)!="DF2SELR1"||vocabulary_!=262144||hidden_!=384||rank_!=32||topk_!=16)throw std::runtime_error("Native selector contract differs");
    projection_.resize(size_t(rank_)*hidden_);predecessors_.resize(size_t(vocabulary_)*rank_);successors_.resize(predecessors_.size());
    for(auto* vector:{&projection_,&predecessors_,&successors_}) {
      f.read(reinterpret_cast<char*>(vector->data()),vector->size()*4);
      if(!f)throw std::runtime_error("Native selector truncated");
      for(float value:*vector)if(!std::isfinite(value))throw std::runtime_error("Nonfinite selector weight");
    }
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Native selector trailing bytes");
  }
  std::vector<int64_t> propose(const float*logits,const float*hidden,int64_t anchor,size_t rows=3,const std::vector<uint32_t>*active=nullptr,bool compact=false)const {
    if(anchor<0||anchor>=vocabulary_||!rows||rows>7)throw std::runtime_error("Invalid selector request");
    using Item=std::pair<float,uint32_t>;
    auto better=[](const Item&a,const Item&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);};
    std::vector<int64_t> result;std::vector<float> context(rank_);
    for(size_t row=0;row<rows;++row){
      std::vector<Item> heap;heap.reserve(topk_);size_t count=active?active->size():vocabulary_;
      if(count<topk_)throw std::runtime_error("Insufficient selector candidates");
      for(size_t i=0;i<count;++i){uint32_t token=active?(*active)[i]:uint32_t(i);if(token>=vocabulary_)throw std::runtime_error("Invalid active selector ID");
        size_t stride=compact&&active?active->size():vocabulary_;
        Item item{logits[row*stride+(compact&&active?i:token)],token};if(!std::isfinite(item.first))continue;
        if(heap.size()<topk_){heap.push_back(item);std::push_heap(heap.begin(),heap.end(),better);}
        else if(better(item,heap.front())){std::pop_heap(heap.begin(),heap.end(),better);heap.back()=item;std::push_heap(heap.begin(),heap.end(),better);}
      }
      if(heap.size()!=topk_)throw std::runtime_error("Insufficient finite selector logits");
      std::sort(heap.begin(),heap.end(),better);
      for(size_t r=0;r<rank_;++r)context[r]=dot(hidden+row*hidden_,projection_.data()+r*hidden_,hidden_)*predecessors_[size_t(anchor)*rank_+r];
      float best=-INFINITY;int64_t next=-1;
      for(const auto& item:heap){float score=item.first+dot(context.data(),successors_.data()+size_t(item.second)*rank_,rank_);
        if(score>best){best=score;next=item.second;}}
      if(next<0)throw std::runtime_error("Invalid selector score");result.push_back(next);anchor=next;
    }
    return result;
  }
};
}
