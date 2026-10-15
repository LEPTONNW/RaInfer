// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include "pte_target.h"
#include <mobile_dflash2/preprocess.h>
#include <mobile_dflash2/cpu_head.h>
#include <mobile_dflash2/cached_body.h>
#include <mobile_dflash2/cpu_selector.h>
#include <mobile_dflash2/hybrid_draft.h>
#include "burst_vote.h"
#include "vision_bundle.h"
#include <nlohmann/json.hpp>
using json=nlohmann::json;
class PortableDraft:public mobile_dflash2::Draft {
  PteTarget& target_;
  std::unique_ptr<et::extension::Module> module_;
  std::vector<float> features_=std::vector<float>(512*6144);
  std::vector<int64_t> tokens_,contexts_=std::vector<int64_t>(512),queries_;
  std::unique_ptr<bool[]> valid_=std::make_unique<bool[]>(512);
  std::vector<TensorPtr> tensors_;
  std::vector<EValue> values_;
  bool uses_qnn_=false;
  std::unique_ptr<et::extension::Module> selector_;
  std::unique_ptr<et::extension::Module> embedding_;
  std::unique_ptr<mobile_dflash2::CpuHead> cpu_head_;
  std::unique_ptr<mobile_dflash2::CpuSelector> native_selector_;
  std::unique_ptr<mobile_dflash2::CachedBody> cpu_body_;
  std::vector<float> cpu_logits_;
  std::vector<uint32_t> active_vocabulary_;
  std::vector<uint32_t> base_vocabulary_;
  unsigned cpu_workers_=4;
  bool compact_=false;
  bool compact_logits_=false;
  bool gpu_head_=false;
  double gpu_setup_seconds_=0;
  double theta_=10000;
  std::vector<float> embeddings_,qcos_,qsin_,kcos_,ksin_,mask_;
  double body_seconds_=0,selector_seconds_=0,packing_seconds_=0,head_seconds_=0;
 public:
  PortableDraft(PteTarget& target,const std::string& path,const json& manifest,const std::string& manifest_path,
      std::unique_ptr<et::extension::Module> preloaded={}):target_(target),
      module_(manifest.value("backend",std::string())=="executorch-cpu-cached"?nullptr:
        preloaded?std::move(preloaded):std::make_unique<et::extension::Module>(path,et::extension::Module::LoadMode::Mmap)),
      tokens_(target.block_width(),4),queries_(target.block_width()) {
    const auto& config=manifest.at("config");
    compact_=manifest.value("input_contract",std::string())=="external-embedding-rope-mask";
    theta_=config.value("rope_theta",10000.0);
    uses_qnn_=manifest.at("backend")=="executorch-qnn-hybrid";
    bool cached=manifest.at("backend")=="executorch-cpu-cached";
    if(manifest.at("algorithm")!="mobile-dflash2-derived"||(!uses_qnn_&&!cached&&manifest.at("backend")!="executorch-cpu-portable")||
       config.at("block_size").get<size_t>()!=target.block_width()||config.at("context_window")!=512||
       config.at("target_hidden_size")!=1536||config.at("hidden_size")!=384||config.at("head_dim")!=64||
       config.at("vocab_size")!=262144||config.at("mask_token_id")!=4||!(theta_>0)||!std::isfinite(theta_)||
       config.at("feature_layers")!=json::array({3,13,23,33})||!target.has_features())
      throw std::runtime_error("DFlash2 target/draft feature contract mismatch");
    if(cached) {
      if(config.at("layers")!=3||config.at("intermediate_size")!=1536||config.at("kv_heads")!=2||
        config.at("conv_kernel_size")!=2||config.at("conv_group_size")!=16||theta_!=10000.0)
        throw std::runtime_error("Unsupported cached CPU architecture");
      auto separator=manifest_path.find_last_of("/\\");auto folder=manifest_path.substr(0,separator+1);
      if(manifest.at("embedding_file")!="embedding-cpu-fp32.pte"||
        (manifest.at("selector_file")!="selector-cpu-fp32.pte"&&manifest.at("selector_file")!="selector-cpu-fp32.bin")||manifest.at("head_file")!="head-cpu-int8.bin")
        throw std::runtime_error("CPU cache companions differ");
      cpu_body_=std::make_unique<mobile_dflash2::CachedBody>(path);
      if(manifest.contains("active_vocabulary_file")) {
        if(manifest.at("active_vocabulary_file")!="active-vocabulary.u32")throw std::runtime_error("Unsupported active vocabulary file");
        std::ifstream f(folder+"active-vocabulary.u32",std::ios::binary|std::ios::ate);auto bytes=f.tellg();
        if(!f||bytes<64||bytes>262144*4||bytes%4)throw std::runtime_error("Active vocabulary file size differs");
        active_vocabulary_.resize(size_t(bytes)/4);f.seekg(0);f.read(reinterpret_cast<char*>(active_vocabulary_.data()),bytes);
        if(!f||!std::is_sorted(active_vocabulary_.begin(),active_vocabulary_.end())||
          std::adjacent_find(active_vocabulary_.begin(),active_vocabulary_.end())!=active_vocabulary_.end()||active_vocabulary_.back()>=262144)
          throw std::runtime_error("Active vocabulary is not a strict valid set");
      }
      cpu_head_=std::make_unique<mobile_dflash2::CpuHead>(folder+"head-cpu-int8.bin");
      if(cpu_head_->hidden_size()!=384||cpu_head_->vocabulary()!=262144)throw std::runtime_error("Cached CPU head dimensions differ");
      embedding_=std::make_unique<et::extension::Module>(folder+"embedding-cpu-fp32.pte",et::extension::Module::LoadMode::Mmap);
      if(manifest.at("selector_file")=="selector-cpu-fp32.bin")native_selector_=std::make_unique<mobile_dflash2::CpuSelector>(folder+"selector-cpu-fp32.bin");
      else {selector_=std::make_unique<et::extension::Module>(folder+"selector-cpu-fp32.pte",et::extension::Module::LoadMode::Mmap);check(selector_->load_method("forward"),"load CPU cached selector");}
      check(embedding_->load_method("forward"),"load CPU cached embedding");
      cpu_logits_.resize(target.block_width()*262144);
      base_vocabulary_=active_vocabulary_;
      return;
    }
    if(uses_qnn_&&(!manifest.value("contains_npu_partitions",false)||manifest.at("qnn_delegates").empty()))
      throw std::runtime_error("Hybrid draft has no declared QNN partitions");
    if(uses_qnn_&&(manifest.value("output_contract",std::string())=="logits-hidden"||manifest.value("output_contract",std::string())=="hidden-cpu-int8-head")) {
      if(manifest.at("selector_file")!="selector-cpu-fp32.pte")throw std::runtime_error("Unsupported companion selector file");
      auto separator=manifest_path.find_last_of("/\\");
      auto folder=separator==std::string::npos?std::string():manifest_path.substr(0,separator+1);
      if(manifest.value("output_contract",std::string())=="hidden-cpu-int8-head") {
        if(manifest.at("head_file")!="head-cpu-int8.bin")throw std::runtime_error("Unsupported CPU head file");
        cpu_head_=std::make_unique<mobile_dflash2::CpuHead>(folder+"head-cpu-int8.bin");
        if(cpu_head_->hidden_size()!=384||cpu_head_->vocabulary()!=262144)throw std::runtime_error("CPU head dimensions changed");
        cpu_logits_.resize(target.block_width()*262144);
      }
      selector_=std::make_unique<et::extension::Module>(folder+"selector-cpu-fp32.pte",et::extension::Module::LoadMode::Mmap);
      if(compact_) {
        if(manifest.at("embedding_file")!="embedding-cpu-fp32.pte")throw std::runtime_error("Unsupported embedding companion");
        embedding_=std::make_unique<et::extension::Module>(folder+"embedding-cpu-fp32.pte",et::extension::Module::LoadMode::Mmap);
      }
      auto selector_meta=selector_->method_meta("forward");
      if(!selector_meta.ok()||selector_meta->num_inputs()!=3||selector_meta->num_outputs()!=1)throw std::runtime_error("Selector method contract mismatch");
    }
    if(compact_&&!embedding_)throw std::runtime_error("Compact draft requires QNN body and CPU companions");
    auto meta=module_->method_meta("forward");if(!meta.ok()||meta->num_inputs()!=(compact_?7:5)||meta->num_outputs()!=(selector_&&!cpu_head_?2:1))
      throw std::runtime_error("Portable draft method contract mismatch");
    size_t width=target.block_width();
    tensors_.push_back(et::extension::make_tensor_ptr({1,512,6144},features_.data()));
    if(compact_) {
      embeddings_.resize(width*384);qcos_.resize(width*64);qsin_.resize(width*64);
      kcos_.resize((512+width)*64);ksin_.resize((512+width)*64);mask_.resize(width*(512+width));
      tensors_.push_back(et::extension::make_tensor_ptr({1,static_cast<int32_t>(width),384},embeddings_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,1,static_cast<int32_t>(width),64},qcos_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,1,static_cast<int32_t>(width),64},qsin_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,1,static_cast<int32_t>(512+width),64},kcos_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,1,static_cast<int32_t>(512+width),64},ksin_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,1,static_cast<int32_t>(width),static_cast<int32_t>(512+width)},mask_.data()));
    } else {
      tensors_.push_back(et::extension::make_tensor_ptr({1,static_cast<int32_t>(width)},tokens_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,512},contexts_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,static_cast<int32_t>(width)},queries_.data()));
      tensors_.push_back(et::extension::make_tensor_ptr({1,512},valid_.get()));
    }
    for(size_t i=0;i<tensors_.size();++i) {
      auto expected=meta->input_tensor_meta(i).get();auto shape=expected.sizes();
      if(tensors_[i]->scalar_type()!=expected.scalar_type()||tensors_[i]->dim()!=shape.size())throw std::runtime_error("Draft tensor type/rank mismatch");
      for(size_t axis=0;axis<shape.size();++axis)if(tensors_[i]->size(axis)!=shape[axis])throw std::runtime_error("Draft tensor shape mismatch");
      values_.emplace_back(*tensors_[i]);
    }
    check(module_->load_method("forward"),"load draft body");
    if(selector_)check(selector_->load_method("forward"),"load selector");
    if(embedding_)check(embedding_->load_method("forward"),"load embedding");
  }
  bool uses_qnn()const{return uses_qnn_;}
  void reset(unsigned workers=4,unsigned body_workers=1,bool compact_logits=false,bool full_vocabulary=false,bool gpu_head=false,bool gpu_check=false){
    if(!workers||workers>4)throw std::runtime_error("CPU workers must be 1..4");
    cpu_workers_=workers;active_vocabulary_=base_vocabulary_;body_seconds_=selector_seconds_=packing_seconds_=head_seconds_=0;
    compact_logits_=compact_logits&&bool(native_selector_);if(full_vocabulary)active_vocabulary_.clear();
    if(gpu_head&&!native_selector_)throw std::runtime_error("GPU head requires native selector");
    gpu_head_=gpu_head;compact_logits_|=gpu_head;
    auto started=std::chrono::steady_clock::now();if(cpu_head_)cpu_head_->use_gpu(gpu_head,gpu_check);
    gpu_setup_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    if(cpu_body_){cpu_body_->set_workers(body_workers);cpu_body_->reset();}
  }
  size_t active_vocabulary_size()const{return active_vocabulary_.size();}
  void add_prompt_tokens(const std::vector<int64_t>& prompt){
    if(active_vocabulary_.empty())return;
    for(auto id:prompt){if(id<0||id>=262144)throw std::runtime_error("Dynamic vocabulary token invalid");active_vocabulary_.push_back(uint32_t(id));}
    std::sort(active_vocabulary_.begin(),active_vocabulary_.end());active_vocabulary_.erase(std::unique(active_vocabulary_.begin(),active_vocabulary_.end()),active_vocabulary_.end());
  }
  std::string backend_name()const{return cpu_body_?(gpu_head_?"cpu-cached-context-fp32-vulkan-int8-head":"cpu-cached-context-fp32-int8-head"):uses_qnn_?"qnn-hybrid":"cpu-portable-fp32";}
  std::string gpu_name()const{return cpu_head_?cpu_head_->gpu_name():std::string();}
  double gpu_setup_seconds()const{return gpu_setup_seconds_;}
  bool gpu_verified()const{return cpu_head_&&cpu_head_->gpu_verified();}
  size_t projected_feature_rows()const{return cpu_body_?cpu_body_->projected_feature_rows():0;}
  void prime_context(){if(cpu_body_)cpu_body_->update(target_.position(),[&](size_t p){return target_.history_features(p);});}
  double body_seconds()const{return body_seconds_;}
  double selector_seconds()const{return selector_seconds_;}
  double packing_seconds()const{return packing_seconds_;}
  double head_seconds()const{return head_seconds_;}
  std::vector<int64_t> propose(int64_t anchor,size_t position)override {
    if(position!=target_.position())throw std::runtime_error("Pending anchor position mismatch");
    if(cpu_body_) {
      auto start=std::chrono::steady_clock::now();
      cpu_body_->update(position,[&](size_t p){return target_.history_features(p);});
      std::fill(tokens_.begin(),tokens_.end(),4);tokens_[0]=anchor;
      if(!active_vocabulary_.empty()){
        auto it=std::lower_bound(active_vocabulary_.begin(),active_vocabulary_.end(),uint32_t(anchor));
        if(it==active_vocabulary_.end()||*it!=uint32_t(anchor))active_vocabulary_.insert(it,uint32_t(anchor));
      }
#if defined(RABI_DFLASH_EXPERIMENTAL_WIDTHS)
      // The packed lookup artifact has a qualified fixed [1,4] input. There
      // are only two distinct IDs here: anchor and mask. Reuse its exact mask
      // embedding for wider diagnostic blocks, without exporting new weights.
      std::vector<int64_t> lookup_ids={anchor,4,4,4};
      auto ids=et::extension::make_tensor_ptr({1,4},lookup_ids.data());
      auto embedded=embedding_->forward({EValue(*ids)});if(!embedded.ok()||embedded->size()!=1)throw std::runtime_error("Cached CPU embedding failed");
      auto input=(*embedded)[0].toTensor();if(input.scalar_type()!=ScalarType::Float||input.numel()!=4*384)throw std::runtime_error("Cached CPU embedding shape mismatch");
      std::vector<float> wider_embedding;
      const float* body_input=input.const_data_ptr<float>();
      if(tokens_.size()!=4){
        wider_embedding.resize(tokens_.size()*384);
        std::copy_n(body_input,384,wider_embedding.data());
        for(size_t r=1;r<tokens_.size();++r)std::copy_n(body_input+384,384,wider_embedding.data()+r*384);
        body_input=wider_embedding.data();
      }
      auto packed=std::chrono::steady_clock::now();packing_seconds_+=std::chrono::duration<double>(packed-start).count();
      auto hidden=cpu_body_->forward(body_input,position,tokens_.size());
#else
      auto ids=et::extension::make_tensor_ptr({1,static_cast<int32_t>(tokens_.size())},tokens_.data());
      auto embedded=embedding_->forward({EValue(*ids)});if(!embedded.ok()||embedded->size()!=1)throw std::runtime_error("Cached CPU embedding failed");
      auto input=(*embedded)[0].toTensor();if(input.scalar_type()!=ScalarType::Float||input.numel()!=tokens_.size()*384)throw std::runtime_error("Cached CPU embedding shape mismatch");
      auto packed=std::chrono::steady_clock::now();packing_seconds_+=std::chrono::duration<double>(packed-start).count();
      auto hidden=cpu_body_->forward(input.const_data_ptr<float>(),position);
#endif
      auto body_end=std::chrono::steady_clock::now();body_seconds_+=std::chrono::duration<double>(body_end-packed).count();
      if(compact_logits_) {
        const auto* active=active_vocabulary_.empty()?nullptr:&active_vocabulary_;
        const size_t stride=active?active->size():262144;
        cpu_logits_.resize((tokens_.size()-1)*stride);
        cpu_head_->project(hidden.data()+384,tokens_.size()-1,cpu_logits_.data(),cpu_workers_,active,true);
        auto head_end=std::chrono::steady_clock::now();head_seconds_+=std::chrono::duration<double>(head_end-body_end).count();
        auto output=native_selector_->propose(cpu_logits_.data(),hidden.data()+384,anchor,tokens_.size()-1,active,true);
        selector_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-head_end).count();return output;
      }
      cpu_logits_.resize(tokens_.size()*262144);
      std::fill_n(cpu_logits_.data(),262144,0.f);cpu_head_->project(hidden.data()+384,tokens_.size()-1,cpu_logits_.data()+262144,cpu_workers_,
        active_vocabulary_.empty()?nullptr:&active_vocabulary_);
      auto head_end=std::chrono::steady_clock::now();head_seconds_+=std::chrono::duration<double>(head_end-body_end).count();
      auto logits=et::extension::make_tensor_ptr({1,static_cast<int32_t>(tokens_.size()),262144},cpu_logits_.data());
      if(native_selector_) {
        auto output=native_selector_->propose(cpu_logits_.data()+262144,hidden.data()+384,anchor,tokens_.size()-1,
          active_vocabulary_.empty()?nullptr:&active_vocabulary_);
        selector_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-head_end).count();return output;
      }
      auto states=et::extension::make_tensor_ptr({1,static_cast<int32_t>(tokens_.size()),384},hidden.data());auto known=et::extension::make_tensor_ptr({1},&anchor);
      auto result=selector_->forward({EValue(*logits),EValue(*states),EValue(*known)});
      if(!result.ok()||result->size()!=1)throw std::runtime_error("Cached CPU selector failed");
      auto tensor=(*result)[0].toTensor();if(tensor.scalar_type()!=ScalarType::Long||tensor.numel()!=tokens_.size()-1)throw std::runtime_error("Cached CPU proposal shape mismatch");
      selector_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-head_end).count();
      auto values=tensor.const_data_ptr<int64_t>();std::vector<int64_t> output(values,values+tensor.numel());
      for(auto value:output)if(value<0||value>=262144)throw std::runtime_error("Cached CPU proposal out of range");return output;
    }
    auto packed_start=std::chrono::steady_clock::now();
    target_.draft_context(features_.data(),contexts_.data(),valid_.get());
    std::fill(tokens_.begin(),tokens_.end(),4);tokens_[0]=anchor;
    for(size_t i=0;i<queries_.size();++i)queries_[i]=position+i;
    if(compact_) {
      auto ids=et::extension::make_tensor_ptr({1,static_cast<int32_t>(tokens_.size())},tokens_.data());
      auto embedded=embedding_->forward({EValue(*ids)});
      if(!embedded.ok()||embedded->size()!=1)throw std::runtime_error("CPU embedding failed");
      auto tensor=(*embedded)[0].toTensor();
      if(tensor.scalar_type()!=ScalarType::Float||tensor.numel()!=embeddings_.size())throw std::runtime_error("Embedding output changed");
      std::memcpy(embeddings_.data(),tensor.const_data_ptr<float>(),embeddings_.size()*sizeof(float));
      mobile_dflash2::prepare_external(contexts_.data(),valid_.get(),queries_.data(),queries_.size(),theta_,
        qcos_.data(),qsin_.data(),kcos_.data(),ksin_.data(),mask_.data());
    }
    auto body_start=std::chrono::steady_clock::now();
    packing_seconds_+=std::chrono::duration<double>(body_start-packed_start).count();
    auto result=module_->forward(values_);if(!result.ok())throw std::runtime_error("Portable draft execution failed");
    auto body_end=std::chrono::steady_clock::now();
    body_seconds_+=std::chrono::duration<double>(body_end-body_start).count();
    std::vector<EValue> selected;
    if(selector_) {
      auto hidden=(*result)[cpu_head_?0:1].toTensor();
      TensorPtr cpu_logits_tensor;
      if(cpu_head_) {
        if(hidden.scalar_type()!=ScalarType::Float||hidden.numel()!=queries_.size()*384)throw std::runtime_error("CPU head input mismatch");
        std::fill_n(cpu_logits_.data(),262144,0.0f);
        cpu_head_->project(hidden.const_data_ptr<float>()+384,queries_.size()-1,cpu_logits_.data()+262144);
        cpu_logits_tensor=et::extension::make_tensor_ptr({1,static_cast<int32_t>(queries_.size()),262144},cpu_logits_.data());
      }
      auto logits=cpu_head_?*cpu_logits_tensor:(*result)[0].toTensor();
      if(logits.scalar_type()!=ScalarType::Float||hidden.scalar_type()!=ScalarType::Float||
         logits.numel()!=queries_.size()*262144||hidden.numel()!=queries_.size()*384)
        throw std::runtime_error("Hybrid body output contract mismatch");
      auto known=et::extension::make_tensor_ptr({1},&anchor);
      auto proposals=selector_->forward({EValue(logits),EValue(hidden),EValue(*known)});
      if(!proposals.ok())throw std::runtime_error("CPU selector execution failed");
      selector_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-body_end).count();
      selected=std::move(proposals.get());
    } else selected=std::move(result.get());
    auto tensor=selected[0].toTensor();
    if(tensor.scalar_type()!=ScalarType::Long||tensor.numel()!=queries_.size()-1)throw std::runtime_error("Invalid draft output");
    auto ids=tensor.const_data_ptr<int64_t>();std::vector<int64_t> candidates(ids,ids+tensor.numel());
    for(auto id:candidates)if(id<0||id>=262144)throw std::runtime_error("Draft token out of range");
    return candidates;
  }
};
static json run_request(PteTarget& target,PortableDraft& draft,const std::vector<int64_t>& prompt,
    size_t budget,bool diagnostic,bool ngram,unsigned workers,bool burst=false,unsigned body_workers=1,bool fast_argmax=false,bool profile=false,bool compact_logits=false,bool full_vocabulary=false,bool fixed_affinity=false,bool gpu_head=false,bool gpu_check=false,bool record_blocks=false) {
    const auto width=target.block_width();
    if(prompt.empty()||prompt.size()>target.verification_capacity()||!budget||budget>(diagnostic?512:2048))
      throw std::runtime_error("Invalid request budget or compiled prompt capacity");
    for(auto id:prompt)if(id<0||id>=262144)throw std::runtime_error("Prompt token invalid");
    mobile_dflash2::CpuWorkers::fixed_affinity(fixed_affinity);mobile_dflash2::CpuWorkers::pin_owner();
    target.configure(fast_argmax,profile,record_blocks);target.reset();draft.reset(workers,body_workers,compact_logits,full_vocabulary,gpu_head,gpu_check);draft.add_prompt_tokens(prompt);
    budget=std::min(budget,target.verification_capacity()-prompt.size()+1);
    int64_t pending=0;
    auto start=std::chrono::steady_clock::now();
    for(size_t offset=0;offset<prompt.size();offset+=width) {
      std::vector<int64_t> block(width,0);auto live=std::min(width,prompt.size()-offset);
      std::copy_n(prompt.begin()+offset,live,block.begin());
      auto result=target.stage(block);pending=result.target_next[live-1];target.commit(live);
    }
    target.assert_image_prefill();
    auto prime_start=std::chrono::steady_clock::now();draft.prime_context();
    double prime_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prime_start).count();
    auto prefilled=std::chrono::steady_clock::now();auto prefill_execution=target.execution_seconds();
    target.clear_decode_profile();
    const auto eos=[](int64_t token){return token==1||token==106;};
    mobile_dflash2::Stats stats;std::vector<int64_t> output{pending};
    mobile_dflash2::HybridDraft selected(draft,prompt,pending,width-1,ngram);
    if(!eos(pending)&&budget>1) {
      auto rest=mobile_dflash2::generate(target,selected,pending,prompt.size(),budget-1,eos,stats);
      output.insert(output.end(),rest.begin(),rest.end());
    }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-prefilled).count();
    return {{"actual_hexagon_execution",true},{"diagnostic_only",diagnostic},{"quality_validated",false},
      {"dflash2_active",selected.neural_rounds>0},{"speculative_active",stats.rounds>0},{"draft_backend",draft.backend_name()},
      {"draft_active_vocabulary_size",draft.active_vocabulary_size()},{"cpu_workers",workers},{"persistent_cpu_pool",true},
      {"draft_contains_executed_npu_partitions",draft.uses_qnn()&&selected.neural_rounds>0},{"block_width",width},
      {"prompt_tokens",prompt.size()},{"tokens",output},{"rounds",stats.rounds},{"proposed_draft_tokens",stats.proposed},
      {"accepted_draft_tokens",stats.accepted},{"committed_tokens",stats.emitted},{"decode_seconds",elapsed},
      {"draft_body_seconds",draft.body_seconds()},{"draft_selector_seconds",draft.selector_seconds()},
      {"draft_head_seconds",draft.head_seconds()},{"draft_body_workers",body_workers},{"fast_argmax",fast_argmax},
      {"compact_logits",compact_logits},{"full_vocabulary",full_vocabulary},
      {"fixed_affinity",fixed_affinity},
      {"gpu_head",gpu_head},{"draft_gpu_device",draft.gpu_name()},{"draft_gpu_setup_seconds",draft.gpu_setup_seconds()},
      {"draft_gpu_cpu_logit_parity",draft.gpu_verified()},
      {"target_timing",target.timing()},
      {"draft_context_pack_seconds",draft.packing_seconds()},
      {"verification_execution_seconds",target.execution_seconds()-prefill_execution},{"explicit_burst_vote",burst},
      {"ngram_enabled",ngram},{"ngram_rounds",selected.ngram_rounds},{"ngram_accepted",selected.ngram_accepted},
      {"ngram_proposed",selected.ngram_proposed},{"neural_draft_rounds",selected.neural_rounds},
      {"finish_reason",eos(output.back())?"stop":"length"},
      {"draft_projected_feature_rows",draft.projected_feature_rows()},{"prefill_draft_context_seconds",prime_seconds},
      {"decode_tokens_per_second",elapsed>0?stats.emitted/elapsed:0.0},
      {"prefill_seconds",std::chrono::duration<double>(prefilled-start).count()},{"vision",target.image_report()}};
}
static void protocol(const json& message){std::cout<<"@DFLASH2 "<<message.dump()<<std::endl;}
int main(int argc,char**argv) {
  try {
    et::runtime::runtime_init();
    if(argc==6&&std::string(argv[1])=="--serve") {
      auto loaded=std::chrono::steady_clock::now();
      std::ifstream m(argv[4]);auto manifest=json::parse(m);
      std::ifstream c(argv[5]);auto contract=json::parse(c);
      PteTarget target(argv[2],contract);
#if defined(RABI_DFLASH_EXPERIMENTAL_WIDTHS)
      if(target.block_width()!=4){
        const char* experimental=std::getenv("RABI_EXPERIMENTAL_WIDTH");
        if(!experimental||std::stoul(experimental)!=target.block_width()||target.block_width()<5||target.block_width()>6)
          throw std::runtime_error("Persistent worker requires qualified AR4 target or explicit experimental width");
      }
#else
      if(target.block_width()!=4)throw std::runtime_error("Persistent worker requires qualified AR4 target");
#endif
      PortableDraft draft(target,argv[3],manifest,argv[4]);
      protocol({{"event","ready"},{"protocol",1},{"context",target.context_capacity()},
        {"vision_bundle_protocol",target.supports_vision_bundle()?1:0},
        {"load_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-loaded).count()}});
      std::string line;size_t requests=0;
      while(std::getline(std::cin,line)) {
        uint64_t id=0;
        try {
          if(line.size()>65536)throw std::runtime_error("Worker request too large");
          auto request=json::parse(line);id=request.at("id").get<uint64_t>();
          if(request.value("command",std::string())=="shutdown")break;
          bool burst=request.value("burst",false);BurstVote vote;if(burst)vote.enable();
          std::vector<int64_t> request_tokens;
          json vision_timing=json::object();
          target.set_images(nullptr);
          if(request.contains("vision_bundle")) {
            if(request.contains("tokens"))throw std::runtime_error("Use tokens or vision_bundle, not both");
            auto bundle=VisionBundle::load(request.at("vision_bundle").get<std::string>());
            request_tokens=std::move(bundle.tokens);target.set_images(std::move(bundle.images));vision_timing=std::move(bundle.encoder_timing);
          } else request_tokens=request.at("tokens").get<std::vector<int64_t>>();
          auto report=run_request(target,draft,request_tokens,
            request.at("max_new_tokens").get<size_t>(),false,request.value("ngram",true),request.value("cpu_workers",4u),
            burst,request.value("body_workers",1u),request.value("fast_argmax",false),request.value("profile",false),
            request.value("compact_logits",false),request.value("full_vocabulary",false),request.value("fixed_affinity",false),request.value("gpu_head",false),request.value("gpu_check",false),request.value("record_blocks",false));
          report["persistent_worker"]=true;report["request_index"]=++requests;
          if(!vision_timing.empty())report["vision_encoder_timing"]=vision_timing;
          target.set_images(nullptr);
          protocol({{"event","result"},{"id",id},{"report",report}});
        }catch(const std::exception& error){target.abort();target.set_images(nullptr);protocol({{"event","error"},{"id",id},{"message",error.what()}});}
      }
      return 0;
    }
    if(argc<9||argc>11||(std::string(argv[8])!="--diagnostic"&&std::string(argv[8])!="--app"))throw std::runtime_error("Usage: mobile_runner TARGET.pte DRAFT.pte MANIFEST.json TOKENS.txt MAX_NEW REPORT.json IO-CONTRACT.json --diagnostic|--app [--draft-first] [--burst-vote]");
    const bool diagnostic=std::string(argv[8])=="--diagnostic";
    bool draft_first=false,burst=false;
    for(int i=9;i<argc;++i){auto flag=std::string(argv[i]);if(flag=="--draft-first")draft_first=true;else if(flag=="--burst-vote")burst=true;else throw std::runtime_error("Unknown diagnostic option");}
    std::ifstream m(argv[3]);auto manifest=json::parse(m);
    std::ifstream input(argv[4]);std::vector<int64_t> prompt;int64_t id;
    while(input>>id){if(id<0||id>=262144)throw std::runtime_error("Prompt token invalid");prompt.push_back(id);}
    if(prompt.empty()||!input.eof())throw std::runtime_error("Invalid prompt file");
    size_t budget=std::stoul(argv[5]);if(!budget||budget>(diagnostic?512:2048))throw std::runtime_error("Invalid output budget");
    std::ifstream c(argv[7]);auto io_contract=json::parse(c);
    std::unique_ptr<et::extension::Module> preloaded;
    if(draft_first&&manifest.at("backend")=="executorch-cpu-cached")throw std::runtime_error("CPU cached draft has no QNN preload");
    if(draft_first) {
      preloaded=std::make_unique<et::extension::Module>(argv[2],et::extension::Module::LoadMode::Mmap);
      check(preloaded->load_method("forward"),"preload draft body");
    }
    PteTarget target(argv[1],io_contract);auto width=target.block_width();
#if defined(RABI_DFLASH_EXPERIMENTAL_WIDTHS)
    if(width!=4&&width!=5&&width!=6&&width!=8)throw std::runtime_error("DFlash2 requires a true multi-row verifier");
#else
    if(width!=4&&width!=8)throw std::runtime_error("DFlash2 requires a true multi-row verifier");
#endif
    PortableDraft draft(target,argv[2],manifest,argv[3],std::move(preloaded));
    BurstVote vote;if(burst)vote.enable();
    auto report=run_request(target,draft,prompt,budget,diagnostic,false,4,burst);
    std::ofstream file(argv[6]);file<<report.dump(2);if(!file)throw std::runtime_error("Report write failed");
    std::cout<<report.dump()<<'\n';
  } catch(const std::exception&error){
    if(argc>1&&std::string(argv[1])=="--serve")protocol({{"event","error"},{"id",0},{"message",error.what()}});
    std::cerr<<error.what()<<'\n';return 1;
  }
}
