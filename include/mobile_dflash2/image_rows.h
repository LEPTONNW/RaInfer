// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mobile_dflash2 {
struct ImageSpan { size_t start, count, embedding_row; };
class ImageRows {
    std::vector<int32_t> row_map_;
    std::vector<float> embeddings_;
    size_t image_rows_ = 0;
public:
    static constexpr size_t hidden_size = 1536;
    static constexpr int64_t placeholder_id = 258880;
    void assign(size_t prompt_tokens, std::vector<float> embeddings, const std::vector<ImageSpan>& spans) {
        if (!prompt_tokens || prompt_tokens > 2044 || embeddings.empty() || embeddings.size() % hidden_size)
            throw std::invalid_argument("Invalid image bundle dimensions");
        const size_t rows = embeddings.size() / hidden_size;
        if (rows > 1024) throw std::invalid_argument("Too many image rows");
        for (float value : embeddings)
            if (!std::isfinite(value)) throw std::invalid_argument("Non-finite image embedding");
        std::vector<int32_t> mapping(prompt_tokens, -1);
        std::vector<bool> used(rows, false);
        size_t mapped = 0;
        for (auto span : spans) {
            if (!span.count || span.start > prompt_tokens || span.count > prompt_tokens - span.start ||
                span.embedding_row > rows || span.count > rows - span.embedding_row)
                throw std::invalid_argument("Image span exceeds prompt or embedding data");
            for (size_t i = 0; i < span.count; ++i) {
                if (mapping[span.start + i] != -1) throw std::invalid_argument("Overlapping image spans");
                if (used[span.embedding_row + i]) throw std::invalid_argument("Duplicate projected image row");
                used[span.embedding_row + i] = true;
                mapping[span.start + i] = int32_t(span.embedding_row + i);
                ++mapped;
            }
        }
        if (mapped != rows) throw std::invalid_argument("Image rows and mapped positions differ");
        row_map_ = std::move(mapping);
        embeddings_ = std::move(embeddings);
        image_rows_ = mapped;
    }
    size_t image_rows() const { return image_rows_; }
    const float* at(size_t position) const {
        if (position >= row_map_.size() || row_map_[position] < 0) return nullptr;
        return embeddings_.data() + size_t(row_map_[position]) * hidden_size;
    }
    size_t count(size_t position, size_t count) const {
        size_t result = 0;
        for (size_t i = 0; i < count; ++i) if (at(position + i)) ++result;
        return result;
    }
    void safe_indices(const int64_t* input, size_t count, size_t position, int64_t* output) const {
        for (size_t i = 0; i < count; ++i) output[i] = at(position + i) ? 0 : input[i];
    }
    void override_unscaled(float* output, size_t rows, size_t position) const {
        // QNN's original first graph scales token embeddings by sqrt(1536).
        // mtmd already returns final projected image vectors, so compensate here.
        const float scale = std::sqrt(float(hidden_size));
        for (size_t row = 0; row < rows; ++row) if (const float* image = at(position + row))
            for (size_t channel = 0; channel < hidden_size; ++channel)
                output[row * hidden_size + channel] = image[channel] / scale;
    }
};
} // namespace mobile_dflash2
