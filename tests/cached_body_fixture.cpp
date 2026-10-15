// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/cached_body.h>
#include <mobile_dflash2/cpu_selector.h>
#include <iostream>
int main(int argc,char**argv){
  try {
    if(argc<5||argc>8)throw std::runtime_error("Usage: cached_body_fixture BODY.bin FIXTURE-DIR POSITION OUTPUT.f32 [HEAD.bin [SELECTOR.bin [ACTIVE.u32]]]");
    std::string folder=argv[2];size_t position=std::stoul(argv[3]);
    auto read=[&](std::string name,void* data,size_t size){std::ifstream f(folder+"/"+name,std::ios::binary);f.read(static_cast<char*>(data),size);if(!f)throw std::runtime_error("Fixture truncated");};
    std::vector<float> features(512*6144),embeddings(4*384);std::vector<int64_t> cp(512);std::vector<uint8_t> valid(512);
    read("features.f32",features.data(),features.size()*4);read("embeddings.f32",embeddings.data(),embeddings.size()*4);
    read("context.i64",cp.data(),cp.size()*8);read("valid.u8",valid.data(),valid.size());
    mobile_dflash2::CachedBody body(argv[1]);
    body.update(position,[&](size_t absolute)->const float*{for(size_t i=0;i<512;++i)if(valid[i]&&size_t(cp[i])==absolute)return features.data()+i*6144;throw std::runtime_error("Future/missing fixture context requested");});
    auto output=body.forward(embeddings.data(),position);auto cached=body.projected_feature_rows();
    body.update(position,[&](size_t)->const float*{throw std::runtime_error("Repeated context was recomputed");});
    auto repeated=body.forward(embeddings.data(),position);
    if(output!=repeated||body.projected_feature_rows()!=cached)throw std::runtime_error("Cache changed repeated inference");
    for(unsigned workers:{2u,3u,4u}){
      mobile_dflash2::CachedBody parallel(argv[1]);parallel.set_workers(workers);
      parallel.update(position,[&](size_t absolute)->const float*{for(size_t i=0;i<512;++i)if(valid[i]&&size_t(cp[i])==absolute)return features.data()+i*6144;throw std::runtime_error("Missing parallel context");});
      if(output!=parallel.forward(embeddings.data(),position))throw std::runtime_error("Parallel body changed hidden states");
    }
    if(position<=512&&position>4) {
      // An intermediate checkpoint must produce the same final cache as one
      // full prefix update. Callbacks reject every future-position access.
      mobile_dflash2::CachedBody incremental(argv[1]);
      auto fill=[&](size_t end){incremental.update(end,[&](size_t absolute)->const float* {
        if(absolute>=end)throw std::runtime_error("Future feature accessed");
        for(size_t i=0;i<512;++i)if(valid[i]&&size_t(cp[i])==absolute)return features.data()+i*6144;
        throw std::runtime_error("Missing incremental feature");
      });};
      fill(position-3);fill(position);
      if(output!=incremental.forward(embeddings.data(),position)||incremental.projected_feature_rows()!=position)
        throw std::runtime_error("Incremental cache differs from a full prefix");
      fill(position-3);fill(position);
      if(output!=incremental.forward(embeddings.data(),position))throw std::runtime_error("Reset/replay cache differs");
    }
    std::ofstream f(argv[4],std::ios::binary);f.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
    if(argc>=6) {
      mobile_dflash2::CpuHead head(argv[5]);std::vector<float> logits(3*head.vocabulary());
      std::vector<uint32_t> active;
      if(argc==8){std::ifstream a(argv[7],std::ios::binary|std::ios::ate);auto size=std::streamoff(a.tellg());if(!a||size<=0||size%4)throw std::runtime_error("Invalid active fixture");
        active.resize(size_t(size)/4);a.seekg(0);a.read(reinterpret_cast<char*>(active.data()),size);if(!a)throw std::runtime_error("Truncated active fixture");}
      if(head.hidden_size()!=384)throw std::runtime_error("Head fixture width differs");
      head.project(output.data()+384,3,logits.data(),4,active.empty()?nullptr:&active);
      std::ofstream h(folder+"/native-logits.f32",std::ios::binary);
      h.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);
      if(!h)throw std::runtime_error("Cannot write native head output");
      if(argc>=7){
        std::vector<int64_t> tokens(4);read("tokens.i64",tokens.data(),32);mobile_dflash2::CpuSelector selector(argv[6]);
        auto ids=selector.propose(logits.data(),output.data()+384,tokens[0],3,active.empty()?nullptr:&active);
        if(!active.empty()){
          std::vector<float> compact(3*active.size());head.project(output.data()+384,3,compact.data(),4,&active,true);
          if(ids!=selector.propose(compact.data(),output.data()+384,tokens[0],3,&active,true))throw std::runtime_error("Compact selector changed draft IDs");
        }
        std::ofstream s(folder+"/native-proposals.i64",std::ios::binary);s.write(reinterpret_cast<const char*>(ids.data()),ids.size()*8);
      }
    }
    std::cout<<"{\"projected_rows\":"<<cached<<",\"repeated_context_reused\":true}\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
