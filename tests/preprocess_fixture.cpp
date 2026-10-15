// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/preprocess.h>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
template<class T>std::vector<T> read(const std::string& path,size_t count) {
 std::vector<T> values(count);std::ifstream f(path,std::ios::binary);
 if(!f.read(reinterpret_cast<char*>(values.data()),count*sizeof(T)))throw std::runtime_error("Fixture read failed");return values;
}
int main(int argc,char**argv) {
 try {
  if(argc!=2)throw std::runtime_error("Fixture folder required");std::string root=argv[1];
  auto cp=read<int64_t>(root+"/context.i64",512),qp=read<int64_t>(root+"/query.i64",4);auto v=read<uint8_t>(root+"/valid.u8",512);
  std::unique_ptr<bool[]> valid(new bool[512]);for(size_t i=0;i<512;++i)valid[i]=v[i]!=0;
  std::vector<std::vector<float>> result={std::vector<float>(4*64),std::vector<float>(4*64),std::vector<float>(516*64),std::vector<float>(516*64),std::vector<float>(4*516)};
  mobile_dflash2::prepare_external(cp.data(),valid.get(),qp.data(),4,10000,result[0].data(),result[1].data(),result[2].data(),result[3].data(),result[4].data());
  for(size_t i=0;i<result.size();++i){std::ofstream f(root+"/native"+std::to_string(i)+".f32",std::ios::binary);f.write(reinterpret_cast<char*>(result[i].data()),result[i].size()*sizeof(float));}
 }catch(const std::exception&e){std::cerr<<e.what();return 1;}
}
