// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <mobile_dflash2/image_rows.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

struct VisionBundle {
    std::vector<int64_t> tokens;
    std::shared_ptr<mobile_dflash2::ImageRows> images;
    nlohmann::json encoder_timing;
    static VisionBundle load(const std::string& filename) {
        namespace fs = std::filesystem;
        std::ifstream stream(filename);
        auto info = nlohmann::json::parse(stream);
        if (info.at("format") != "rainfer-vision-v1" || info.at("hidden_size") != 1536 ||
            info.at("dtype") != "float32-le" || info.at("image_placeholder_id") != 258880)
            throw std::runtime_error("Unsupported image bundle format");
        if (info.at("embedding_semantics") != "projected-unscaled; divide by sqrt(1536) before NPU token-embedding scale")
            throw std::runtime_error("Image embedding scale contract differs");
        const size_t rows = info.at("embedding_rows").get<size_t>();
        if (!rows || rows > 1024) throw std::runtime_error("Invalid image embedding count");
        fs::path name = info.at("embeddings_file").get<std::string>();
        if (name.empty() || name != name.filename() || name == "." || name == "..")
            throw std::runtime_error("Image embedding data must be beside its manifest");
        fs::path binary = fs::path(filename).parent_path() / name;
        if (fs::file_size(binary) != rows * 1536 * sizeof(float))
            throw std::runtime_error("Image embedding file size differs");
        std::vector<float> values(rows * 1536);
        std::ifstream data(binary, std::ios::binary);
        data.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
        if (!data) throw std::runtime_error("Image embedding read failed");
        VisionBundle result;
        result.tokens = info.at("tokens").get<std::vector<int64_t>>();
        if (result.tokens.empty() || result.tokens.size() > 2044)
            throw std::runtime_error("Image prompt exceeds qualified capacity");
        std::vector<mobile_dflash2::ImageSpan> spans;
        for (const auto& item : info.at("spans"))
            spans.push_back({item.at("start").get<size_t>(), item.at("count").get<size_t>(), item.at("embedding_row").get<size_t>()});
        result.images = std::make_shared<mobile_dflash2::ImageRows>();
        result.images->assign(result.tokens.size(), std::move(values), spans);
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            if (result.tokens[i] < 0 || result.tokens[i] >= 262144 ||
                (result.images->at(i) && result.tokens[i] != mobile_dflash2::ImageRows::placeholder_id))
                throw std::runtime_error("Invalid image placeholder or prompt token");
        }
        result.encoder_timing = info.value("timings", nlohmann::json::object());
        return result;
    }
};
