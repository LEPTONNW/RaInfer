// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include "pte_target.h"
#include "burst_vote.h"
#include "vision_bundle.h"
int main(int argc,char** argv) {
  try {
    et::runtime::runtime_init();
    if(argc<6)throw std::runtime_error("Usage: pte_probe MODEL.pte TOKENS.txt|BUNDLE.json MAX_NEW OUTPUT.json FEATURES.f32 [IO-CONTRACT.json] [--vision-bundle] [--prefill-logits OUTPUT.u16] [--sequential-prefill] [--burst-vote]");
    std::vector<int64_t> prompt;
    int count=std::stoi(argv[3]);if(count<1||count>512)throw std::runtime_error("Invalid output budget");
    nlohmann::json contract;if(argc>=7){std::ifstream c(argv[6]);contract=nlohmann::json::parse(c);}
    bool sequential=false,burst=false,vision=false;std::string logits_path;
    for(int i=7;i<argc;++i){auto flag=std::string(argv[i]);if(flag=="--sequential-prefill")sequential=true;else if(flag=="--burst-vote")burst=true;
      else if(flag=="--vision-bundle")vision=true;
      else if(flag=="--prefill-logits"&&i+1<argc)logits_path=argv[++i];
      else throw std::runtime_error("Unknown probe option");}
    PteTarget target(argv[1],contract);auto width=target.block_width();
    if(vision) {
      if(sequential)throw std::runtime_error("Image probe requires qualified AR4 prefill");
      auto bundle=VisionBundle::load(argv[2]);prompt=std::move(bundle.tokens);target.set_images(std::move(bundle.images));
    } else {
      std::ifstream input(argv[2]);int64_t value;
      while(input>>value){if(value<0||value>=262144)throw std::runtime_error("Token out of range");prompt.push_back(value);}
      if(prompt.empty()||!input.eof())throw std::runtime_error("Invalid prompt token file");
    }
    if(prompt.size()>target.verification_capacity())throw std::runtime_error("Prompt exceeds compiled capacity");
    count=std::min(count,static_cast<int>(target.verification_capacity()-prompt.size()+1));
    BurstVote vote;if(burst)vote.enable();
    std::ofstream features(argv[5],std::ios::binary);if(!features)throw std::runtime_error("Cannot write features");
    std::vector<int64_t> predictions,output;
    auto start=std::chrono::steady_clock::now();
    auto stride=sequential?size_t(1):width;
    for(size_t offset=0;offset<prompt.size();offset+=stride) {
      std::vector<int64_t> block(width,0);auto live=std::min(stride,prompt.size()-offset);
      std::copy_n(prompt.begin()+offset,live,block.begin());auto result=target.stage(block);
      predictions.insert(predictions.end(),result.target_next.begin(),result.target_next.begin()+live);
      target.dump_features(features,live);
      if(!logits_path.empty()&&offset+live==prompt.size())target.dump_native_logits(logits_path);
      target.commit(live);
    }
    target.assert_image_prefill();
    auto prefill=std::chrono::steady_clock::now();
    int64_t pending=predictions.back();
    for(int i=0;i<count;++i) {
      output.push_back(pending);if(pending==1||pending==106||i+1==count)break;
      std::vector<int64_t> block(width,0);block[0]=pending;
      auto result=target.stage(block);target.dump_features(features,1);predictions.push_back(result.target_next[0]);target.commit(1);pending=result.target_next[0];
    }
    double prefill_seconds=std::chrono::duration<double>(prefill-start).count();
    double decode_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prefill).count();
    std::ofstream report(argv[4]);report<<"{\"actual_hexagon_execution\":true,\"dflash2_active\":false,\"explicit_burst_vote\":"<<(burst?"true":"false")<<",\"prompt_tokens\":"<<prompt.size()
      <<",\"prefill_seconds\":"<<prefill_seconds<<",\"decode_seconds\":"<<decode_seconds<<",\"decode_tokens_per_second\":"
      <<(output.size()>1?(output.size()-1)/decode_seconds:0)<<",\"tokens\":[";
    for(size_t i=0;i<output.size();++i){if(i)report<<',';report<<output[i];}report<<"],\"predictions\":[";
    for(size_t i=0;i<predictions.size();++i){if(i)report<<',';report<<predictions[i];}report<<"],\"vision\":"<<target.image_report().dump()<<"}\n";
    if(!report)throw std::runtime_error("Cannot write report");
    std::cout<<"{\"event\":\"probe_completed\",\"tokens\":"<<output.size()<<",\"dflash2_active\":false}\n";
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
