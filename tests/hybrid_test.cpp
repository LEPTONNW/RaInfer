// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/hybrid_draft.h>
#include <mobile_dflash2/cpu_workers.h>
#include <iostream>
#include <atomic>
#include <set>
using namespace mobile_dflash2;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Neural:Draft {size_t calls=0;std::vector<Token> propose(Token,size_t)override{++calls;return {90,91,92};}};
int main(){try {
  std::vector<Token> prompt{1,2,3,4,5,6,7,8,11,12,13,1,2,3,4,5,6,7,8,11,12,13,1,2,3,4,5,6,7};
  Neural neural;HybridDraft hybrid(neural,prompt,8,3,true);
  require(hybrid.propose(8,0)==std::vector<Token>({11,12,13})&&neural.calls==0,"Causal repeated ngram missed");
  hybrid.feedback(0,3,{99});require(hybrid.propose(99,1)==std::vector<Token>({90,91,92})&&neural.calls==1,"Rejected cache did not use neural fallback");
  prompt[19]=21;prompt[20]=22;prompt[21]=23;Neural second;HybridDraft ambiguous(second,prompt,8,3,true);
  require(ambiguous.propose(8,0)==std::vector<Token>({90,91,92}),"Ambiguous continuations selected");
  Neural third;HybridDraft fresh(third,{1,2,3,4,5,6,7},8,3,true);
  require(fresh.propose(8,0)==std::vector<Token>({90,91,92}),"Unknown future was used");
  CpuWorkers pool;std::vector<std::thread::id> first(4),second_ids(4);
  pool.run(4,[&](unsigned id){first[id]=std::this_thread::get_id();});
  pool.run(4,[&](unsigned id){second_ids[id]=std::this_thread::get_id();});
  require(first==second_ids&&std::set<std::thread::id>(first.begin(),first.end()).size()==4,"Worker threads were not reused");
  bool caught=false;try{pool.run(4,[](unsigned id){if(id==2)throw std::runtime_error("fixture");});}catch(const std::runtime_error&){caught=true;}
  require(caught,"Worker failure was swallowed");std::atomic<int> completed{0};
  for(unsigned workers=1;workers<=4;++workers)pool.run(workers,[&](unsigned){++completed;});
  require(completed==10,"Worker pool lost tasks after exception");
  std::cout<<"Causal ngram, ambiguity, fallback, pool identity and exception recovery passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
