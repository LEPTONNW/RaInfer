// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#include <mobile_dflash2/image_rows.h>
#include <cstring>
#include <iostream>
#include <limits>
using namespace mobile_dflash2;
int main() {
    try {
        std::vector<float> values(3 * 1536);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 97) - 48) / 19;
        ImageRows image;
        image.assign(9, values, {{3, 3, 0}}); // deliberately spans an AR4 boundary
        const int64_t ids[4] = {2, 255999, 258880, 258880};
        int64_t safe[4]{};
        image.safe_indices(ids, 4, 1, safe);
        if (safe[0] != 2 || safe[1] != 255999 || safe[2] != 0 || safe[3] != 0)
            throw std::runtime_error("Image PLE padding changed a text boundary token");
        std::vector<float> block(4 * 1536, 17.f), original = block;
        image.override_unscaled(block.data(), 4, 0);
        if (std::memcmp(block.data(), original.data(), 3 * 1536 * sizeof(float)))
            throw std::runtime_error("Text rows changed");
        for (size_t i = 0; i < 1536; ++i)
            if (std::abs(block[3 * 1536 + i] * std::sqrt(1536.f) - values[i]) > 1e-6f)
                throw std::runtime_error("Projected image scaling differs");
        image.override_unscaled(block.data(), 4, 4);
        for (size_t r = 0; r < 2; ++r) for (size_t i = 0; i < 1536; ++i)
            if (std::abs(block[r * 1536 + i] * std::sqrt(1536.f) - values[(r + 1) * 1536 + i]) > 1e-6f)
                throw std::runtime_error("Cross-block image position changed");
        original = block;
        image.override_unscaled(block.data(), 4, 9);
        if (block != original) throw std::runtime_error("Generated text inherited an earlier image");
        bool rejected = false;
        try { ImageRows bad; bad.assign(9, values, {{0, 2, 0}, {1, 1, 2}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Overlapping image spans accepted");
        rejected = false;
        try { ImageRows bad; bad.assign(9, values, {{0, 2, 0}, {4, 1, 1}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Duplicate embedding row accepted");
        values[0] = std::numeric_limits<float>::quiet_NaN(); rejected = false;
        try { ImageRows bad; bad.assign(9, values, {{3, 3, 0}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Non-finite image accepted");
        std::cout << "Image scaling, PLE padding, block boundaries and invalid input checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
