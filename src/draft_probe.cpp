// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include "pte_target.h"
template<class T>std::vector<T> read(const std::string& path,size_t count) {
  std::vector<T> values(count);std::ifstream f(path,std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(values.data()),count*sizeof(T))||f.peek()!=EOF)throw std::runtime_error("Invalid fixture: "+path);
  return values;
}
int main(int argc,char**argv) {
 try {
  et::runtime::runtime_init();
  if(argc!=5)throw std::runtime_error("Usage: draft_probe BODY.pte SELECTOR.pte FIXTURE_DIR REPORT.json");
  std::string root=argv[3];auto features=read<float>(root+"/features.f32",512*6144);
  auto ids=read<int64_t>(root+"/tokens.i64",4),cp=read<int64_t>(root+"/context.i64",512),qp=read<int64_t>(root+"/query.i64",4);
  auto validity=read<uint8_t>(root+"/valid.u8",512);std::unique_ptr<bool[]> valid(new bool[512]);
  for(size_t i=0;i<512;++i)valid[i]=validity[i]!=0;
  std::vector<TensorPtr> tensors={et::extension::make_tensor_ptr({1,512,6144},features.data()),et::extension::make_tensor_ptr({1,4},ids.data()),
    et::extension::make_tensor_ptr({1,512},cp.data()),et::extension::make_tensor_ptr({1,4},qp.data()),et::extension::make_tensor_ptr({1,512},valid.get())};
  std::vector<EValue> inputs;for(auto& tensor:tensors)inputs.emplace_back(*tensor);
  et::extension::Module body(argv[1],et::extension::Module::LoadMode::Mmap),selector(argv[2],et::extension::Module::LoadMode::Mmap);
  std::vector<double> times;std::vector<int64_t> proposed;
  for(int iteration=0;iteration<4;++iteration) {
    auto start=std::chrono::steady_clock::now();auto output=body.forward(inputs);
    if(!output.ok()||output->size()!=2)throw std::runtime_error("Draft body execution failed");
    if(iteration==3)for(size_t index=0;index<2;++index) {
      auto tensor=(*output)[index].toTensor();
      if(tensor.scalar_type()!=ScalarType::Float)throw std::runtime_error("Expected float body output");
      std::ofstream dump(std::string(argv[4])+".output"+std::to_string(index)+".f32",std::ios::binary);
      dump.write(reinterpret_cast<const char*>(tensor.const_data_ptr<float>()),tensor.numel()*sizeof(float));
      if(!dump)throw std::runtime_error("Body output dump failed");
    }
    auto anchor=et::extension::make_tensor_ptr({1},ids.data());
    auto chosen=selector.forward({(*output)[0],(*output)[1],EValue(*anchor)});
    if(!chosen.ok())throw std::runtime_error("Selector execution failed");
    auto t=(*chosen)[0].toTensor();if(t.scalar_type()!=ScalarType::Long||t.numel()!=3)throw std::runtime_error("Proposal contract changed");
    const auto* data=t.const_data_ptr<int64_t>();proposed.assign(data,data+3);
    times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
  nlohmann::json report={{"actual_phone_execution",true},{"backend","qnn-hybrid-body-cpu-selector"},{"proposals",proposed},
    {"seconds_including_first_load",times},{"diagnostic_only",true},{"generation_speed_measurement",false}};
  std::ofstream file(argv[4]);file<<report.dump(2);if(!file)throw std::runtime_error("Report write failed");std::cout<<report.dump()<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
