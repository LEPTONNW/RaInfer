// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <mobile_dflash2/image_rows.h>

struct VisionEmbeddingStats { size_t primary_rows = 0, ple_rows = 0; };
struct VisionEmbeddingInvocation {
    const mobile_dflash2::ImageRows* images = nullptr;
    size_t position = 0;
    VisionEmbeddingStats* stats = nullptr;
};
extern thread_local VisionEmbeddingInvocation rainfer_vision_invocation;
class VisionEmbeddingScope {
    VisionEmbeddingInvocation previous_;
public:
    VisionEmbeddingScope(const mobile_dflash2::ImageRows* images, size_t position, VisionEmbeddingStats* stats)
        : previous_(rainfer_vision_invocation) { rainfer_vision_invocation = {images, position, stats}; }
    ~VisionEmbeddingScope() { rainfer_vision_invocation = previous_; }
    VisionEmbeddingScope(const VisionEmbeddingScope&) = delete;
};
