// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/cpu_head.h>
#include <iostream>
int main() {
  try {
    const uint32_t vocabulary=37,hidden=32;std::vector<int8_t> weights(vocabulary*hidden);
    std::vector<float> scales(vocabulary),queries(3*hidden),actual(3*vocabulary);
    for(size_t i=0;i<weights.size();++i)weights[i]=int8_t(int(i*31%255)-127);
    for(size_t i=0;i<scales.size();++i)scales[i]=.001f*(i+1);
    for(size_t r=0;r<3;++r)for(size_t c=0;c<hidden;++c)queries[r*hidden+c]=c==0?127.f:float(int(c*13+r*7)%255-127);
    std::string path="cpu-head-test.bin";
    {std::ofstream file(path,std::ios::binary);file.write("DF2HEAD1",8);file.write(reinterpret_cast<const char*>(&vocabulary),4);
     file.write(reinterpret_cast<const char*>(&hidden),4);file.write(reinterpret_cast<const char*>(scales.data()),scales.size()*4);
     file.write(reinterpret_cast<const char*>(weights.data()),weights.size());}
    mobile_dflash2::CpuHead head(path);head.project(queries.data(),3,actual.data(),4);
    for(size_t r=0;r<3;++r)for(size_t v=0;v<vocabulary;++v) {
      int32_t dot=0;for(size_t c=0;c<hidden;++c)dot+=int32_t(queries[r*hidden+c])*int32_t(weights[v*hidden+c]);
      float expected=float(dot)*scales[v];if(std::abs(expected-actual[r*vocabulary+v])>1e-4)throw std::runtime_error("Signed INT8 projection differs from scalar reference");
    }
    std::vector<uint32_t> active{2,7,19,36};std::vector<float> selected(actual.size());
    head.project(queries.data(),3,selected.data(),4,&active);
    std::vector<float> compact(3*active.size());head.project(queries.data(),3,compact.data(),4,&active,true);
    for(size_t r=0;r<3;++r)for(size_t i=0;i<active.size();++i)
      if(compact[r*active.size()+i]!=actual[r*vocabulary+active[i]])throw std::runtime_error("Compact head changed a logit");
    for(size_t r=0;r<3;++r)for(size_t v=0;v<vocabulary;++v) {
      bool wanted=std::find(active.begin(),active.end(),v)!=active.end();
      if(wanted?selected[r*vocabulary+v]!=actual[r*vocabulary+v]:selected[r*vocabulary+v]!=-INFINITY)
        throw std::runtime_error("Selected vocabulary projection differs or leaks an inactive row");
    }
    std::cout<<"CPU head signed quantization, selected vocabulary and threaded projection passed\n";
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
