// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <executorch/runtime/core/event_tracer.h>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include "qnn_probe_event.h"
// Optional host spans, including RPC/runtime overhead; never tensor/text logs.
class TargetProfile final:public executorch::runtime::EventTracer {
  using R=executorch::runtime::EventTracerEntry;
  struct Metric {std::string name;size_t calls=0;double seconds=0;};
  std::vector<Metric> metrics_;
  std::map<std::string,size_t> names_;
  bool enabled_=false;
  std::map<std::string,nlohmann::json> qnn_,memory_,thread_,copies_;
  static int64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
public:
  void enable(bool value){enabled_=value;clear();}
  void clear(){for(auto& m:metrics_){m.calls=0;m.seconds=0;}qnn_.clear();memory_.clear();thread_.clear();copies_.clear();}
  nlohmann::json thread_snapshot()const {nlohmann::json result=nlohmann::json::array();for(const auto& item:thread_)result.push_back(item.second);return result;}
  nlohmann::json copy_snapshot()const {nlohmann::json result=nlohmann::json::array();for(const auto& item:copies_)result.push_back(item.second);return result;}
  nlohmann::json qnn_snapshot()const {nlohmann::json result=nlohmann::json::array();for(const auto& item:qnn_)result.push_back(item.second);return result;}
  nlohmann::json memory_snapshot()const {nlohmann::json result=nlohmann::json::array();for(const auto& item:memory_)result.push_back(item.second);return result;}
  nlohmann::json snapshot()const {
    nlohmann::json result=nlohmann::json::array();
    for(const auto& m:metrics_)if(m.calls)result.push_back({{"name",m.name},{"calls",m.calls},{"seconds",m.seconds}});
    return result;
  }
  void create_event_block(const char*)override{}
  R start_profiling(const char* name,executorch::runtime::ChainID chain=-1,executorch::runtime::DebugHandle handle=0)override {
    if(!enabled_)return {-1,chain,handle,0,executorch::runtime::DelegateDebugIdType::kNone};
    std::string key=name?name:"unnamed";
    if(key=="QnnBackend"||key=="Delegate"||key=="DELEGATE_CALL"||key=="QNN_BIND"||key=="BOUNDARY_MEMCPY")key+="/"+std::to_string(current_debug_handle());
    auto found=names_.find(key);size_t index;
    if(found==names_.end()){index=metrics_.size();names_.emplace(key,index);metrics_.push_back({key});}else index=found->second;
    return {int64_t(index),chain,handle,et_timestamp_t(now()),executorch::runtime::DelegateDebugIdType::kNone};
  }
  void end_profiling(R entry)override {
    if(entry.event_id<0)return;
    auto& m=metrics_.at(size_t(entry.event_id));++m.calls;m.seconds+=double(now()-entry.start_time)*1e-9;
  }
  R start_profiling_delegate(const char* name,executorch::runtime::DelegateDebugIntId index)override{return start_profiling(name,-1,uint32_t(index));}
  void end_profiling_delegate(R entry,const void* =nullptr,size_t =0)override{end_profiling(entry);}
  void log_profiling_delegate(const char* name,executorch::runtime::DelegateDebugIntId,et_timestamp_t,et_timestamp_t,const void* metadata=nullptr,size_t bytes=0)override {
    if(!enabled_||!metadata)return;
    const auto graph=current_debug_handle();
    if(bytes==sizeof(QnnProbeEvent)&&static_cast<const QnnProbeEvent*>(metadata)->magic==0x514e4e50){
      const auto& e=*static_cast<const QnnProbeEvent*>(metadata);
      std::string key=std::to_string(graph)+"/"+std::to_string(e.type)+"/"+std::to_string(e.unit)+"/"+(name?name:"");
      if(qnn_.size()>8192&&!qnn_.count(key))return;
      auto& row=qnn_[key];if(row.is_null())row={{"graph_instruction",graph},{"name",name?name:""},{"type",e.type},{"unit",e.unit},{"parent_type",e.parent},{"depth",e.depth},{"calls",0u},{"sum",0ull},{"minimum",e.value},{"maximum",e.value}};
      row["calls"]=row["calls"].get<uint64_t>()+1;row["sum"]=row["sum"].get<uint64_t>()+e.value;
      row["minimum"]=std::min(row["minimum"].get<uint64_t>(),e.value);row["maximum"]=std::max(row["maximum"].get<uint64_t>(),e.value);
    } else if(bytes==sizeof(QnnProbeMemory)&&static_cast<const QnnProbeMemory*>(metadata)->magic==0x4d454d50){
      const auto& e=*static_cast<const QnnProbeMemory*>(metadata);auto& row=memory_[std::to_string(graph)];
      row={{"graph_instruction",graph},{"input_shared",e.input_shared},{"input_raw",e.input_raw},{"output_shared",e.output_shared},{"output_raw",e.output_raw},{"shared_bytes",e.shared_bytes},{"raw_bytes",e.raw_bytes}};
    } else if(bytes==sizeof(QnnProbeThread)&&static_cast<const QnnProbeThread*>(metadata)->magic==0x43505550){
      const auto& e=*static_cast<const QnnProbeThread*>(metadata);auto& row=thread_[std::to_string(graph)];
      if(row.is_null())row={{"graph_instruction",graph},{"calls",0u},{"wall_ns",0ull},{"cpu_ns",0ull}};
      row["calls"]=row["calls"].get<uint64_t>()+1;row["wall_ns"]=row["wall_ns"].get<uint64_t>()+e.wall_ns;row["cpu_ns"]=row["cpu_ns"].get<uint64_t>()+e.cpu_ns;
    } else if(bytes==sizeof(QnnProbeCopy)&&static_cast<const QnnProbeCopy*>(metadata)->magic==0x434f5059){
      const auto& e=*static_cast<const QnnProbeCopy*>(metadata);auto& row=copies_[std::to_string(graph)];
      if(row.is_null())row={{"instruction",graph},{"calls",0u},{"bytes",0ull},{"identical_pointer_calls",0u}};
      row["calls"]=row["calls"].get<uint64_t>()+1;row["bytes"]=row["bytes"].get<uint64_t>()+e.bytes;row["identical_pointer_calls"]=row["identical_pointer_calls"].get<uint64_t>()+e.identical;
    }
  }
  void track_allocation(executorch::runtime::AllocatorID,size_t)override{}
  executorch::runtime::AllocatorID track_allocator(const char*)override{return 0;}
  executorch::runtime::Result<bool> log_evalue(const executorch::runtime::EValue&,executorch::runtime::LoggedEValueType)override{return false;}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(const char*,executorch::runtime::DelegateDebugIntId,const executorch::aten::Tensor&)override{return false;}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(const char*,executorch::runtime::DelegateDebugIntId,const executorch::runtime::ArrayRef<executorch::aten::Tensor>)override{return false;}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(const char*,executorch::runtime::DelegateDebugIntId,const int&)override{return false;}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(const char*,executorch::runtime::DelegateDebugIntId,const bool&)override{return false;}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(const char*,executorch::runtime::DelegateDebugIntId,const double&)override{return false;}
  void set_delegation_intermediate_output_filter(executorch::runtime::EventTracerFilterBase*)override{}
};
