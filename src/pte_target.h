// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
// Independent persistent ExecuTorch/QNN contract probe. No app replacement.
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/platform/runtime.h>
#include <mobile_dflash2/engine.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>
#include "target_profile.h"
#include <mobile_dflash2/argmax.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include "planned_memory.h"
#include "vision_embedding.h"

namespace et=executorch;
using et::runtime::EValue;
using et::aten::ScalarType;
using et::extension::TensorPtr;
static void check(et::runtime::Error error,const char* action) {
  if(error!=et::runtime::Error::Ok)throw std::runtime_error(std::string(action)+": "+std::to_string(static_cast<int>(error)));
}
struct Buffer {
  void* address=nullptr; size_t bytes=0;TensorPtr tensor;
  Buffer(const et::runtime::TensorInfo& info,void* base,size_t offset):bytes(info.nbytes()) {
    address=static_cast<std::byte*>(base)+offset;
    std::memset(address,0,bytes);
    QnnExecuTorchAddCustomMemTensorAddr(address,base);
    auto s=info.sizes();std::vector<et::aten::SizesType> sizes(s.begin(),s.end());
    tensor=et::extension::make_tensor_ptr(std::move(sizes),address,{}, {},info.scalar_type());
  }
  ~Buffer(){tensor.reset();}
  Buffer(const Buffer&)=delete;
};
class PteTarget final:public mobile_dflash2::Target {
  struct RpcPool {
    void* address=nullptr;
    ~RpcPool(){if(address)QnnExecuTorchFreeCustomMem(address);}
  } pool_;
  PlannedRpcMemory planned_;
  // Buffers declared first: module is destroyed before RPC memory is freed.
  std::vector<std::unique_ptr<Buffer>> inputs_,outputs_;
  TargetProfile profile_;
  et::extension::Module module_;
  std::vector<EValue> args_;
  size_t width_=0,context_=0,past_=0,position_=0;
  bool staged_=false;
  double execution_seconds_=0;
  double prepare_seconds_=0,argmax_seconds_=0,commit_seconds_=0;
  bool fast_argmax_=false;
  bool record_blocks_=false;
  nlohmann::json blocks_=nlohmann::json::array();
  int qnn_profile_level_=0;
  std::shared_ptr<mobile_dflash2::ImageRows> image_rows_;
  VisionEmbeddingStats image_stats_;
  bool vision_supported_=false;
  float logits_scale_=0;
  std::vector<float> history_=std::vector<float>(512*6144);
  std::vector<std::pair<float,int>> feature_encodings_;
  std::vector<std::pair<float,int>> cache_encodings_;
 public:
  bool supports_vision_bundle() const { return vision_supported_; }
  void set_images(std::shared_ptr<mobile_dflash2::ImageRows> images) {
    if(images&&!vision_supported_)throw std::runtime_error("This PTE/IO contract is text-only; select the vision-qualified model");
    image_rows_=std::move(images);image_stats_={};
  }
  void assert_image_prefill() const {
    if(image_rows_&&(image_stats_.primary_rows!=image_rows_->image_rows()||image_stats_.ple_rows!=image_rows_->image_rows()))
      throw std::runtime_error("Image vectors did not reach both target embedding boundaries; use the vision-qualified PTE");
  }
  nlohmann::json image_report() const {return {{"image_tokens",image_rows_?image_rows_->image_rows():0},
    {"primary_rows_injected",image_stats_.primary_rows},{"ple_rows_padded",image_stats_.ple_rows}};}
  explicit PteTarget(const std::string& path,const nlohmann::json& contract={}):module_(path,et::extension::Module::LoadMode::Mmap) {
    if(contract.contains("vision"))vision_supported_=contract.at("vision").value("operator",std::string())=="rainfer::embedding_4bit.dtype_out";
    if(const char* level=std::getenv("RABI_QNN_PROFILE_LEVEL"))qnn_profile_level_=std::stoi(level);
    if(qnn_profile_level_<0||qnn_profile_level_>2)throw std::runtime_error("Invalid QNN profile level");
    et::runtime::BackendOptions<1> options;check(options.set_option(QNN_RUNTIME_PROFILE_LEVEL,qnn_profile_level_),"set QNN profile level");
    check(et::runtime::set_option(QNN_BACKEND,options.view()),"configure QNN runtime");
    if(contract.contains("outputs")&&!contract.at("outputs").empty())logits_scale_=contract.at("outputs").at(0).value("scale",0.f);
    auto m=module_.method_meta("kv_forward");if(!m.ok())throw std::runtime_error("Missing kv_forward method");
    auto meta=m.get();
    if(meta.num_inputs()!=34||(meta.num_outputs()!=31&&meta.num_outputs()!=35))
      throw std::runtime_error("Unsupported Gemma E2B I/O contract");
    // Load persistent graph weights before reserving/registering application
    // I/O buffers. This gives the large output head first access to DSP maps.
    if(const char* shared=std::getenv("RABI_QNN_SHARED_PLANNED"))if(std::string(shared)=="1")planned_.prepare(path,meta);
    check(module_.load_method("kv_forward",planned_.allocator.get(),&profile_),"load target method");
    size_t total=0;
    for(size_t i=0;i<meta.num_inputs();++i)total+=(meta.input_tensor_meta(i).get().nbytes()+127)&~size_t(127);
    for(size_t i=0;i<meta.num_outputs();++i)total+=(meta.output_tensor_meta(i).get().nbytes()+127)&~size_t(127);
    pool_.address=QnnExecuTorchAllocCustomMem(total,128);
    if(!pool_.address)throw std::runtime_error("QNN RPC pool allocation failed");
    size_t offset=0;
    for(size_t i=0;i<meta.num_inputs();++i) {
      auto info=meta.input_tensor_meta(i);if(!info.ok())throw std::runtime_error("Non-tensor input");
      inputs_.push_back(std::make_unique<Buffer>(info.get(),pool_.address,offset));args_.emplace_back(*inputs_.back()->tensor);
      offset+=(info.get().nbytes()+127)&~size_t(127);
    }
    for(size_t i=0;i<meta.num_outputs();++i) {
      auto info=meta.output_tensor_meta(i);if(!info.ok())throw std::runtime_error("Non-tensor output");
      outputs_.push_back(std::make_unique<Buffer>(info.get(),pool_.address,offset));
      offset+=(info.get().nbytes()+127)&~size_t(127);
      check(module_.set_output("kv_forward",EValue(*outputs_.back()->tensor),i),"bind output");
    }
    width_=inputs_[0]->tensor->size(1);context_=inputs_[1]->tensor->size(3);past_=context_-width_;
    if(inputs_[0]->tensor->scalar_type()!=ScalarType::Long||inputs_[3]->tensor->scalar_type()!=ScalarType::Int)
      throw std::runtime_error("Token/position scalar contract changed");
    for(size_t i=4;i<34;++i) {
      auto type=inputs_[i]->tensor->scalar_type();
      if(type!=ScalarType::UInt16&&type!=ScalarType::Float)throw std::runtime_error("Unsupported KV cache type");
      if(outputs_[i-3]->tensor->scalar_type()!=ScalarType::UInt16)throw std::runtime_error("Expected KV16 output");
      if(type==ScalarType::Float) {
        if(!contract.contains("outputs")||contract.at("outputs").size()<31)throw std::runtime_error("Float KV input requires output affine contract");
        const auto& encoding=contract.at("outputs").at(i-3);
        float scale=encoding.at("scale").get<float>();int zero=encoding.at("zero_point").get<int>();
        if(!(scale>0)||!std::isfinite(scale)||zero<0||zero>65535)throw std::runtime_error("Invalid KV output affine encoding");
        cache_encodings_.push_back({scale,zero});
      } else cache_encodings_.push_back({1.0f,0});
    }
    if(inputs_[1]->tensor->scalar_type()!=ScalarType::UInt16||inputs_[2]->tensor->scalar_type()!=ScalarType::UInt16)
      throw std::runtime_error("Expected quantized U16 mask");
    if(has_features()) {
      if(!contract.contains("features")||contract.at("features").size()!=4||contract.at("rows").get<size_t>()!=width_)
        throw std::runtime_error("Feature I/O affine contract is required");
      for(size_t layer=0;layer<4;++layer) {
        const auto& feature=contract.at("features").at(layer);auto type=outputs_[31+layer]->tensor->scalar_type();
        if(feature.at("index")!=31+layer)throw std::runtime_error("Feature order changed");
        if(type==ScalarType::UInt16) {
          if(feature.at("dtype")!="torch.uint16")throw std::runtime_error("Feature scalar contract changed");
          auto scale=feature.at("scale").get<float>();auto zero=feature.at("zero_point").get<int>();
          if(!(scale>0)||!std::isfinite(scale)||zero<0||zero>65535)throw std::runtime_error("Feature affine encoding invalid");
          feature_encodings_.push_back({scale,zero});
        } else if(type==ScalarType::Float) {
          if(feature.at("dtype")!="torch.float32")throw std::runtime_error("Feature scalar contract changed");
          feature_encodings_.push_back({1.0f,0});
        } else throw std::runtime_error("Unsupported auxiliary state type");
      }
    }
    std::cout<<"{\"event\":\"target_loaded\",\"rows\":"<<width_<<",\"context\":"<<context_
             <<",\"outputs\":"<<outputs_.size()<<",\"dflash2_active\":false}\n";
  }
  size_t block_width()const override{return width_;}
  size_t context_capacity()const{return context_;}
  size_t verification_capacity()const{return past_;}
  bool can_verify()const override{return !staged_&&position_+width_<=past_;}
  size_t position()const{return position_;}
  bool has_features()const{return outputs_.size()==35;}
  const float* history_features(size_t absolute)const {
    if(!has_features()||absolute>=position_||position_-absolute>512)throw std::runtime_error("History access would leave the committed prefix");
    return history_.data()+(absolute%512)*6144;
  }
  double execution_seconds()const{return execution_seconds_;}
  void configure(bool fast_argmax,bool profile,bool record_blocks=false){fast_argmax_=fast_argmax;profile_.enable(profile);record_blocks_=record_blocks;}
  void clear_decode_profile(){profile_.clear();prepare_seconds_=argmax_seconds_=commit_seconds_=0;blocks_=nlohmann::json::array();}
  nlohmann::json timing()const{return {{"prepare_seconds",prepare_seconds_},{"argmax_seconds",argmax_seconds_},{"commit_seconds",commit_seconds_},{"operators",profile_.snapshot()},
    {"qnn_events",profile_.qnn_snapshot()},{"memory_bindings",profile_.memory_snapshot()},{"qnn_profile_level",qnn_profile_level_},
    {"api_thread_times",profile_.thread_snapshot()},{"boundary_copies",profile_.copy_snapshot()},
    {"planned_rpc_bytes",planned_.bytes},{"planned_rpc_tensor_starts",planned_.registered},{"verification_blocks",blocks_}};}
  void reset() {
    if(staged_)throw std::runtime_error("Cannot reset a staged KV transaction");
    for(auto& buffer:inputs_)std::memset(buffer->address,0,buffer->bytes);
    std::fill(history_.begin(),history_.end(),0.0f);
    position_=0;execution_seconds_=0;
    clear_decode_profile();
  }
  void draft_context(float* features,int64_t* positions,bool* valid)const {
    if(!has_features())throw std::runtime_error("Target has no auxiliary states");
    size_t count=std::min<size_t>(512,position_),start=position_-count;
    std::fill(features,features+512*6144,0.0f);
    std::fill(valid,valid+512,false);std::fill(positions,positions+512,0);
    for(size_t i=0;i<count;++i) {
      size_t destination=512-count+i,absolute=start+i;
      std::memcpy(features+destination*6144,history_.data()+(absolute%512)*6144,6144*sizeof(float));
      positions[destination]=absolute;valid[destination]=true;
    }
  }
  mobile_dflash2::Verification stage(const std::vector<mobile_dflash2::Token>& tokens)override {
    auto prepare_start=std::chrono::steady_clock::now();
    if(staged_||tokens.size()!=width_||position_+width_>past_)throw std::runtime_error("Invalid/overflowed KV transaction");
    auto ids=static_cast<int64_t*>(inputs_[0]->address);auto positions=static_cast<int32_t*>(inputs_[3]->address);
    auto causal=static_cast<uint16_t*>(inputs_[1]->address);auto sliding=static_cast<uint16_t*>(inputs_[2]->address);
    // Pinned mask encoding: real 0 -> 65535, masked -65535 -> 0.
    for(size_t row=0;row<width_;++row) {
      ids[row]=tokens[row];positions[row]=position_+row;
      for(size_t column=0;column<context_;++column) {
        bool live=(column<position_)||(column>=past_&&column<=past_+row);
        size_t absolute=column<past_?column:position_+column-past_;
        causal[row*context_+column]=live?65535:0;
        sliding[row*context_+column]=(live&&absolute+512>position_+row)?65535:0;
      }
    }
    auto started=std::chrono::steady_clock::now();
    prepare_seconds_+=std::chrono::duration<double>(started-prepare_start).count();
    VisionEmbeddingScope vision_scope(image_rows_.get(),position_,&image_stats_);
    auto result=module_.execute("kv_forward",args_);
    if(!result.ok())throw std::runtime_error("QNN execution failed: "+std::to_string(static_cast<int>(result.error())));
    execution_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    if(result->size()!=outputs_.size())throw std::runtime_error("Runtime output count changed");
    auto selection_start=std::chrono::steady_clock::now();
    auto tensor=(*result)[0].toTensor();auto vocabulary=tensor.size(2);
    mobile_dflash2::Verification verification;
    for(size_t row=0;row<width_;++row) {
      int64_t best=0;
      if(tensor.scalar_type()==ScalarType::UInt16) {
        auto values=tensor.const_data_ptr<uint16_t>()+row*vocabulary;
        best=fast_argmax_?mobile_dflash2::argmax_u16(values,vocabulary):std::max_element(values,values+vocabulary)-values;
      } else if(tensor.scalar_type()==ScalarType::Float) {
        auto values=tensor.const_data_ptr<float>()+row*vocabulary;
        best=std::max_element(values,values+vocabulary)-values;
      } else throw std::runtime_error("Unsupported logits type");
      verification.target_next.push_back(best);
    }
    argmax_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-selection_start).count();
    if(record_blocks_)blocks_.push_back({{"position",position_},{"input",tokens},{"argmax",verification.target_next}});
    staged_=true;return verification;
  }
  void commit(size_t rows)override {
    if(record_blocks_&&!blocks_.empty())blocks_.back()["committed_rows"]=rows;
    auto started=std::chrono::steady_clock::now();
    if(!staged_||rows<1||rows>width_)throw std::runtime_error("Invalid committed prefix");
    for(size_t layer=0;layer<15;++layer) {
      const size_t hd=inputs_[4+layer]->tensor->size(2);
      auto newk=static_cast<uint16_t*>(outputs_[1+layer]->address);
      auto newv=static_cast<uint16_t*>(outputs_[16+layer]->address);
      for(size_t h=0;h<hd;++h)for(size_t r=0;r<rows;++r)write_cache(layer,h*past_+position_+r,newk[h*width_+r]);
      for(size_t r=0;r<rows;++r)for(size_t h=0;h<hd;++h)write_cache(15+layer,(position_+r)*hd+h,newv[r*hd+h]);
    }
    if(has_features())for(size_t layer=0;layer<4;++layer) {
      for(size_t row=0;row<rows;++row)for(size_t channel=0;channel<1536;++channel)
        history_[((position_+row)%512)*6144+layer*1536+channel]=feature_value(layer,row,channel);
    }
    position_+=rows;staged_=false;
    commit_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  }
  void abort()noexcept override {staged_=false;}
  void write_cache(size_t cache,size_t index,uint16_t value) {
    auto& buffer=*inputs_[4+cache];
    if(buffer.tensor->scalar_type()==ScalarType::UInt16)static_cast<uint16_t*>(buffer.address)[index]=value;
    else {
      const auto encoding=cache_encodings_[cache];
      static_cast<float*>(buffer.address)[index]=(static_cast<int>(value)-encoding.second)*encoding.first;
    }
  }
  float feature_value(size_t layer,size_t row,size_t channel)const {
    const auto& output=*outputs_[31+layer];const auto index=row*1536+channel;
    if(output.tensor->scalar_type()==ScalarType::Float)return static_cast<const float*>(output.address)[index];
    const auto encoding=feature_encodings_.at(layer);
    return (static_cast<int>(static_cast<const uint16_t*>(output.address)[index])-encoding.second)*encoding.first;
  }
  void dump_features(std::ofstream& stream,size_t rows)const {
    if(outputs_.size()!=35) return;
    if(!rows||rows>width_)throw std::runtime_error("Invalid feature prefix");
    std::vector<float> values(rows*6144);
    for(size_t row=0;row<rows;++row)for(size_t layer=0;layer<4;++layer)for(size_t channel=0;channel<1536;++channel)
      values[row*6144+layer*1536+channel]=feature_value(layer,row,channel);
    stream.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    if(!stream)throw std::runtime_error("Feature output write failed");
  }
  void dump_native_logits(const std::string& path)const {
    if(!staged_||outputs_[0]->tensor->scalar_type()!=ScalarType::UInt16)
      throw std::runtime_error("Native logits dump requires a staged U16 verifier");
    std::ofstream stream(path,std::ios::binary);
    stream.write(static_cast<const char*>(outputs_[0]->address),outputs_[0]->bytes);
    if(!stream)throw std::runtime_error("Native logits output write failed");
  }
  void dump_soft_targets(std::ofstream& stream,size_t rows,size_t topk,float temperature)const {
    if(!staged_||!rows||rows>width_||topk<2||topk>128||!(temperature>0))throw std::runtime_error("Invalid soft-target request");
    const auto& tensor=*outputs_[0]->tensor;const size_t vocabulary=262144;
    if(tensor.numel()!=width_*vocabulary)throw std::runtime_error("Logits vocabulary differs");
    auto value=[&](size_t row,size_t token)->float {
      if(tensor.scalar_type()==ScalarType::Float)return tensor.const_data_ptr<float>()[row*vocabulary+token];
      if(tensor.scalar_type()!=ScalarType::UInt16||!(logits_scale_>0))throw std::runtime_error("Missing physical logit scale");
      // Softmax is translation invariant; the affine zero point cancels.
      return float(tensor.const_data_ptr<uint16_t>()[row*vocabulary+token])*logits_scale_;
    };
    using Item=std::pair<float,uint32_t>;
    auto better=[](const Item&a,const Item&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);};
    for(size_t row=0;row<rows;++row) {
      std::vector<Item> heap;heap.reserve(topk);float maximum=-INFINITY;
      for(size_t token=0;token<vocabulary;++token) {
        Item item{value(row,token),uint32_t(token)};if(!std::isfinite(item.first))throw std::runtime_error("Nonfinite teacher logit");
        maximum=std::max(maximum,item.first);
        if(heap.size()<topk){heap.push_back(item);std::push_heap(heap.begin(),heap.end(),better);}
        else if(better(item,heap.front())){std::pop_heap(heap.begin(),heap.end(),better);heap.back()=item;std::push_heap(heap.begin(),heap.end(),better);}
      }
      double normalizer=0;for(size_t token=0;token<vocabulary;++token)normalizer+=std::exp(double(value(row,token)-maximum)/temperature);
      double logz=std::log(normalizer),mass=0;std::sort(heap.begin(),heap.end(),better);
      std::vector<uint32_t> ids;std::vector<float> logp;
      for(const auto& item:heap){ids.push_back(item.second);float probability=float(double(item.first-maximum)/temperature-logz);logp.push_back(probability);mass+=std::exp(double(probability));}
      float other=float(std::clamp(1.0-mass,0.0,1.0));
      stream.write(reinterpret_cast<const char*>(ids.data()),topk*4);stream.write(reinterpret_cast<const char*>(logp.data()),topk*4);
      stream.write(reinterpret_cast<const char*>(&other),4);
      if(!stream)throw std::runtime_error("Soft target output write failed");
    }
  }
};
