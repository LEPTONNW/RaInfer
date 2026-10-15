// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
// Delegates original text dequantization to the existing ExecuTorch kernel.
#include "vision_embedding.h"
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <array>

namespace et = executorch;
thread_local VisionEmbeddingInvocation rainfer_vision_invocation;

static void vision_embedding(et::runtime::KernelRuntimeContext& context, et::runtime::Span<et::runtime::EValue*> stack) {
    auto original = et::runtime::get_op_function_from_registry("quantized_decomposed::embedding_4bit.dtype_out");
    if (!original.ok()) throw std::runtime_error("Original quantized embedding kernel is not registered");
    const auto invocation = rainfer_vision_invocation;
    if (!invocation.images) { original.get()(context, stack); return; }
    if (stack.size() != 9) throw std::runtime_error("Vision embedding operator contract changed");
    auto& indices = stack[5]->toTensor();
    if (indices.scalar_type() != et::aten::ScalarType::Long || indices.numel() != 4)
        throw std::runtime_error("Vision requires the qualified AR4 index tensor");
    const size_t count = invocation.images->count(invocation.position, 4);
    if (!count) { original.get()(context, stack); return; }
    auto& output = stack[7]->toTensor();
    if (output.scalar_type() != et::aten::ScalarType::Float || output.dim() != 3 || output.size(1) != 4 ||
        (output.size(2) != 1536 && output.size(2) != 8960))
        throw std::runtime_error("Unsupported Gemma primary/PLE embedding output");
    std::array<int64_t, 4> safe{};
    invocation.images->safe_indices(indices.const_data_ptr<int64_t>(), 4, invocation.position, safe.data());
    auto tensor = et::extension::make_tensor_ptr({1, 4}, safe.data());
    et::runtime::EValue value(*tensor);
    std::array<et::runtime::EValue*, 9> args{};
    std::copy(stack.begin(), stack.end(), args.begin());
    args[5] = &value;
    original.get()(context, {args.data(), args.size()});
    if (context.failure_state() != et::runtime::Error::Ok) return;
    if (output.size(2) == 1536) {
        invocation.images->override_unscaled(output.mutable_data_ptr<float>(), 4, invocation.position);
        if (invocation.stats) invocation.stats->primary_rows += count;
    } else {
        // The original kernel computed pad-token PLE rows using safe index 0.
        if (invocation.stats) invocation.stats->ple_rows += count;
    }
}

static const et::runtime::Kernel kernel("rainfer::embedding_4bit.dtype_out", vision_embedding);
static const auto registration = et::runtime::register_kernels({&kernel, 1});
