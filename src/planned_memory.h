// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/schema/program_generated.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <set>
// An opt-in RPC arena for the existing PTE memory plan. Tensor shapes, offsets
// and aliasing are unchanged; register only actual tensor start addresses.
struct PlannedRpcMemory {
  void* base=nullptr;size_t bytes=0,registered=0;
  std::vector<executorch::runtime::Span<uint8_t>> spans;
  std::unique_ptr<executorch::runtime::HierarchicalAllocator> allocator;
  ~PlannedRpcMemory(){allocator.reset();if(base)QnnExecuTorchFreeCustomMem(base);}
  void prepare(const std::string& path,const executorch::runtime::MethodMeta& meta){
    for(size_t i=0;i<meta.num_memory_planned_buffers();++i)bytes+=(size_t(meta.memory_planned_buffer_size(i).get())+127)&~size_t(127);
    if(!bytes||bytes>1024ull*1024*1024)throw std::runtime_error("Unexpected planned RPC arena size");
    base=QnnExecuTorchAllocCustomMem(bytes,128);if(!base)throw std::runtime_error("Planned RPC allocation failed");
    std::memset(base,0,bytes);size_t offset=0;
    for(size_t i=0;i<meta.num_memory_planned_buffers();++i){size_t size=meta.memory_planned_buffer_size(i).get();spans.emplace_back(static_cast<uint8_t*>(base)+offset,size);offset+=(size+127)&~size_t(127);}
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);struct stat info{};if(fd<0||fstat(fd,&info)!=0){if(fd>=0)close(fd);throw std::runtime_error("Cannot inspect PTE memory plan");}
    void* mapped=mmap(nullptr,info.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(mapped==MAP_FAILED)throw std::runtime_error("Cannot map PTE memory plan");
    try {
      const auto* program=executorch_flatbuffer::GetProgram(mapped);std::set<void*> addresses;
      for(const auto* plan:*program->execution_plan())if(plan->name()->str()=="kv_forward")for(const auto* value:*plan->values()) {
        auto* tensor=value->val_as_Tensor();if(!tensor||!tensor->allocation_info())continue;
        const auto* a=tensor->allocation_info();if(!a->memory_id()||a->memory_id()>spans.size())throw std::runtime_error("Planned memory ID differs");
        uint64_t o=uint64_t(a->memory_offset_low())|(uint64_t(a->memory_offset_high())<<32);auto& span=spans[a->memory_id()-1];
        if(o>=span.size())throw std::runtime_error("Planned tensor outside RPC arena");
        auto* address=span.data()+o;if(addresses.insert(address).second)QnnExecuTorchAddCustomMemTensorAddr(address,base);
      }
      registered=addresses.size();if(!registered)throw std::runtime_error("No planned tensors registered");
    }catch(...){munmap(mapped,info.st_size);throw;}
    munmap(mapped,info.st_size);
    allocator=std::make_unique<executorch::runtime::HierarchicalAllocator>(executorch::runtime::Span(spans.data(),spans.size()));
  }
};
