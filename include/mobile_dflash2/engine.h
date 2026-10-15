// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace mobile_dflash2 {
using Token = int64_t;
struct Verification {
  // logits argmax after each processed input row: [anchor, drafts...].
  std::vector<Token> target_next;
};
class Target {
 public:
  virtual ~Target() = default;
  virtual size_t block_width() const = 0;
  virtual bool can_verify() const{return true;}
  virtual Verification stage(const std::vector<Token>& input) = 0;
  // Retain only this processed prefix. Remaining staged KV/hidden rows are
  // discarded. The last emitted token is pending and has NOT been processed.
  virtual void commit(size_t processed_rows) = 0;
  virtual void abort() noexcept = 0;
};
class Draft {
 public:
  virtual ~Draft() = default;
  virtual std::vector<Token> propose(Token pending, size_t position) = 0;
  virtual void feedback(size_t,size_t,const std::vector<Token>&) {}
};
struct Stats {
  size_t rounds = 0, proposed = 0, accepted = 0, emitted = 0;
};
struct Step {
  std::vector<Token> emitted;
  size_t accepted = 0;
  bool eos = false;
};
inline Step verified_step(Target& target, Token anchor,
                          const std::vector<Token>& candidates,
                          size_t budget, const std::function<bool(Token)>& is_eos) {
  if (!budget || candidates.size() + 1 != target.block_width())
    throw std::invalid_argument("Budget or fixed verifier block contract invalid");
  std::vector<Token> input{anchor};
  input.insert(input.end(), candidates.begin(), candidates.end());
  try {
    auto result = target.stage(input);
    if (result.target_next.size() != input.size())
      throw std::runtime_error("Verifier returned incorrect row count");
    size_t matches = 0;
    while (matches < candidates.size() && candidates[matches] == result.target_next[matches]) ++matches;
    Step step;
    for (size_t i = 0; i < matches && step.emitted.size() < budget; ++i) {
      step.emitted.push_back(candidates[i]);
      ++step.accepted;
      if (is_eos(candidates[i])) { step.eos = true; break; }
    }
    if (!step.eos && step.emitted.size() < budget) {
      step.emitted.push_back(result.target_next[matches]);
      step.eos = is_eos(step.emitted.back());
    }
    target.commit(step.emitted.size());
    return step;
  } catch (...) { target.abort(); throw; }
}
inline std::vector<Token> generate(Target& target, Draft& draft, Token pending,
    size_t processed_position, size_t budget, const std::function<bool(Token)>& is_eos,
    Stats& stats) {
  std::vector<Token> output;
  while (output.size() < budget && !is_eos(pending) && target.can_verify()) {
    auto candidates = draft.propose(pending, processed_position);
    auto step = verified_step(target, pending, candidates, budget-output.size(), is_eos);
    draft.feedback(step.accepted,candidates.size(),step.emitted);
    ++stats.rounds; stats.proposed += candidates.size(); stats.accepted += step.accepted;
    stats.emitted += step.emitted.size();
    output.insert(output.end(), step.emitted.begin(), step.emitted.end());
    processed_position += step.emitted.size();
    pending = step.emitted.back();
    if (step.eos) break;
  }
  return output;
}
} // namespace mobile_dflash2
