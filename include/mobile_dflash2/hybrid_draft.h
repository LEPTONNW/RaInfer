// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <mobile_dflash2/engine.h>
#include <algorithm>
namespace mobile_dflash2 {
// Prompt + already emitted tokens only. Never import generated future labels.
class HybridDraft:public Draft {
  Draft& neural_;
  std::vector<Token> history_;
  bool enabled_=false,used_ngram_=false;
  unsigned cooldown_=0;
  size_t width_=0;
public:
  size_t ngram_rounds=0,ngram_accepted=0,ngram_proposed=0,neural_rounds=0;
  HybridDraft(Draft& neural,std::vector<Token> prompt,Token pending,size_t width,bool enabled)
    :neural_(neural),history_(std::move(prompt)),enabled_(enabled),width_(width){history_.push_back(pending);}
  std::vector<Token> propose(Token pending,size_t position)override {
    if(history_.empty()||history_.back()!=pending)throw std::runtime_error("N-gram pending/history mismatch");
    used_ngram_=false;
    if(enabled_&&!cooldown_) {
      const size_t count=history_.size(),limit=std::min(count/2,size_t(12));
      for(size_t n=limit;n>=4;--n) {
        std::vector<Token> agreed;size_t matches=0;bool ambiguous=false;
        for(size_t offset=0;offset+n+width_<=count-n;++offset) {
          if(!std::equal(history_.begin()+offset,history_.begin()+offset+n,history_.end()-n))continue;
          std::vector<Token> continuation(history_.begin()+offset+n,history_.begin()+offset+n+width_);
          if(!matches)agreed=continuation;else if(agreed!=continuation){ambiguous=true;break;}
          ++matches;
        }
        if(!ambiguous&&(matches>=2||(n>=8&&matches))) {
          used_ngram_=true;++ngram_rounds;ngram_proposed+=width_;return agreed;
        }
      }
    }
    if(cooldown_)--cooldown_;
    ++neural_rounds;return neural_.propose(pending,position);
  }
  void feedback(size_t accepted,size_t proposed,const std::vector<Token>& emitted)override {
    if(used_ngram_){ngram_accepted+=accepted;if(accepted==0)cooldown_=8;}
    neural_.feedback(accepted,proposed,emitted);
    history_.insert(history_.end(),emitted.begin(),emitted.end());
  }
};
}
