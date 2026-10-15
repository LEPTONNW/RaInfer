// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/argmax.h>
#include <random>
#include <iostream>
int main(){
  try {
    std::mt19937 random(57);
    for(size_t n:{1u,7u,8u,9u,31u,1023u,262144u}){
      std::vector<uint16_t> values(n);
      for(int trial=0;trial<15;++trial){
        for(auto& value:values)value=uint16_t(random()%65535);
        if(trial%3==0){values[n/3]=65535;values[n-1]=65535;}
        if(trial%3==1)std::fill(values.begin(),values.end(),0);
        auto expected=std::max_element(values.begin(),values.end())-values.begin();
        if(mobile_dflash2::argmax_u16(values.data(),n)!=size_t(expected))throw std::runtime_error("Argmax index/tie differs");
      }
    }
    bool rejected=false;try{mobile_dflash2::argmax_u16(nullptr,0);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("Empty logits accepted");
    std::cout<<"Argmax scalar/NEON, tails, duplicate maxima and empty input passed\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
