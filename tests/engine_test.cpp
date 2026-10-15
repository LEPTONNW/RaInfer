// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/engine.h>
#include <iostream>
#include <string>
using namespace mobile_dflash2;
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct FakeTarget : Target {
  std::vector<Token> committed, staged;
  bool active=false, fail=false;
  size_t width=4;
  size_t block_width() const override { return width; }
  Verification stage(const std::vector<Token>& input) override {
    require(!active, "Previous transaction was not closed"); active=true; staged=input;
    if(fail) return {{1}};
    Verification result;
    // Deliberately input-dependent predictions: a wrong draft changes later
    // rows, which must never leak into committed state after first mismatch.
    for (auto t:input) result.target_next.push_back(t+1);
    return result;
  }
  void commit(size_t n) override {
    require(active&&n>0&&n<=staged.size(),"Invalid commit");
    committed.insert(committed.end(),staged.begin(),staged.begin()+n); staged.clear();active=false;
  }
  void abort() noexcept override {staged.clear();active=false;}
};
struct WrongDraft:Draft { std::vector<Token> propose(Token t,size_t) override{return {t+1,t+9,t+10};} };
int main() {
  const auto eos=[](Token t){return t==99;};
  { FakeTarget t; auto s=verified_step(t,10,{11,12,13},100,eos);
    require(s.emitted==std::vector<Token>({11,12,13,14})&&s.accepted==3,"All-accept/bonus");
    require(t.committed==std::vector<Token>({10,11,12,13}),"Bonus must remain pending"); }
  { FakeTarget t; auto s=verified_step(t,10,{80,81,82},100,eos);
    require(s.emitted==std::vector<Token>({11})&&s.accepted==0,"First mismatch");
    require(t.committed==std::vector<Token>({10}),"Reject KV rollback"); }
  { FakeTarget t; auto s=verified_step(t,10,{11,88,89},100,eos);
    require(s.emitted==std::vector<Token>({11,12})&&s.accepted==1,"Partial accept");
    require(t.committed==std::vector<Token>({10,11}),"Partial KV rollback"); }
  { FakeTarget t; auto s=verified_step(t,10,{11,12,13},2,eos);
    require(s.emitted==std::vector<Token>({11,12})&&t.committed==std::vector<Token>({10,11}),"Budget rollback"); }
  { FakeTarget t; auto s=verified_step(t,97,{98,99,100},50,eos);
    require(s.eos&&s.emitted==std::vector<Token>({98,99})&&t.committed==std::vector<Token>({97,98}),"EOS rollback"); }
  { FakeTarget t; t.fail=true; bool caught=false;
    try {verified_step(t,10,{11,12,13},100,eos);}catch(const std::runtime_error&){caught=true;}
    require(caught&&!t.active&&t.committed.empty(),"Failed transaction abort"); }
  { FakeTarget t; WrongDraft d;Stats stats;auto values=generate(t,d,10,5,13,eos,stats);
    for(size_t i=0;i<values.size();++i)require(values[i]==11+static_cast<Token>(i),"Speculation differs from greedy target");
    require(t.committed.back()==22&&values.back()==23,"Final pending token duplicated"); }
  std::cout << "7 speculative transaction tests passed; mock target only, no phone benchmark\n";
}
