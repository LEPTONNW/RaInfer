// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include "pte_target.h"
#include <filesystem>
#include <thread>
using json=nlohmann::json;
int main(int argc,char**argv) {
  try {
    et::runtime::runtime_init();
    if(argc!=5)throw std::runtime_error("Usage: target_capture MODEL.pte IO-CONTRACT.json JOBS.json OUTPUT-DIR");
    std::ifstream cj(argv[2]),jj(argv[3]);auto contract=json::parse(cj);auto jobs=json::parse(jj);
    PteTarget target(argv[1],contract);auto width=target.block_width();
    auto dir=std::filesystem::path(argv[4]);std::filesystem::create_directories(dir);
    for(size_t index=0;index<jobs.size();++index) {
      if(std::filesystem::exists(dir/"STOP"))break;
      auto stem=std::to_string(index);auto rp=dir/(stem+".json");auto fp=dir/(stem+".f32");
      if(std::filesystem::exists(rp))continue;
      const auto& job=jobs.at(index);auto prompt=job.at("tokens").get<std::vector<int64_t>>();
      size_t budget=job.at("max_new_tokens").get<size_t>();
      if(prompt.empty()||!budget||budget>512||prompt.size()+budget+width>=2044)throw std::runtime_error("Invalid capture prompt/budget");
      for(auto token:prompt)if(token<0||token>=262144)throw std::runtime_error("Capture token out of range");
      target.reset();std::ofstream features(fp,std::ios::binary);if(!features)throw std::runtime_error("Cannot write features");
      auto topk=job.value("teacher_top_k",size_t(0));float temperature=job.value("teacher_temperature",1.f);std::ofstream soft;
      if(topk){if(topk<2||topk>128||!std::isfinite(temperature)||temperature<=0)throw std::runtime_error("Invalid teacher distribution controls");
        soft.open(dir/(stem+".soft"),std::ios::binary);uint32_t count=uint32_t(topk);soft.write("DF2SOFT1",8);soft.write(reinterpret_cast<const char*>(&count),4);soft.write(reinterpret_cast<const char*>(&temperature),4);}
      std::vector<int64_t> output,predictions;
      auto start=std::chrono::steady_clock::now();
      for(size_t offset=0;offset<prompt.size();offset+=width) {
        std::vector<int64_t> block(width,0);auto live=std::min(width,prompt.size()-offset);
        std::copy_n(prompt.begin()+offset,live,block.begin());auto result=target.stage(block);
        predictions.insert(predictions.end(),result.target_next.begin(),result.target_next.begin()+live);
        target.dump_features(features,live);if(topk)target.dump_soft_targets(soft,live,topk,temperature);target.commit(live);
      }
      auto prefilled=std::chrono::steady_clock::now();int64_t pending=predictions.back();bool interrupted=false;
      for(size_t i=0;i<budget;++i) {
        output.push_back(pending);if(pending==1||pending==106||i+1==budget)break;
        if(std::filesystem::exists(dir/"STOP")){interrupted=true;break;}
        std::vector<int64_t> block(width,0);block[0]=pending;auto result=target.stage(block);
        target.dump_features(features,1);if(topk)target.dump_soft_targets(soft,1,topk,temperature);predictions.push_back(result.target_next[0]);target.commit(1);pending=result.target_next[0];
      }
      features.close();if(topk)soft.close();if(interrupted)break;
      double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prefilled).count();
      json report={{"index",index},{"id",job.at("id")},{"actual_hexagon_execution",true},{"dflash2_active",false},
        {"prompt_tokens",prompt.size()},{"tokens",output},{"predictions",predictions},{"decode_seconds",seconds},
        {"prefill_seconds",std::chrono::duration<double>(prefilled-start).count()},{"decode_tokens_per_second",output.size()>1?(output.size()-1)/seconds:0},
        {"feature_rows",prompt.size()+output.size()-1},{"feature_channels",6144}};
      if(topk){report["teacher_top_k"]=topk;report["teacher_temperature"]=temperature;}
      auto temporary=dir/(stem+".json.tmp");{std::ofstream f(temporary);f<<report.dump(2);if(!f)throw std::runtime_error("Report write failed");}
      std::filesystem::rename(temporary,rp);
      std::cout<<json({{"event","sample_captured"},{"index",index},{"response_tokens",output.size()}}).dump()<<std::endl;
      if(job.value("wait_for_ack",false)) {
        auto ack_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
        while(!std::filesystem::exists(dir/(stem+".ack"))) {
          if(std::filesystem::exists(dir/"STOP"))return 0;
          if(std::chrono::steady_clock::now()>ack_deadline){std::cerr<<"PC acknowledgement timeout; saved sample preserved\n";return 0;}
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
    }
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
