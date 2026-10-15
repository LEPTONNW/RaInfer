// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
// Standalone mmproj preprocessing/encoding helper. No NPU or host app required.
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <gguf.h>
#include "vision_bundle.h"
#include <mobile_dflash2/cpu_workers.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static std::string read_text(const std::string& filename) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read prompt file");
    return {std::istreambuf_iterator<char>(stream), {}};
}
int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> options;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::runtime_error("Expected --option value pairs");
            if (!options.emplace(argv[i], argv[i + 1]).second) throw std::runtime_error("Duplicate option");
        }
        for (const auto* name : {"--text-model", "--mmproj", "--image", "--prompt-file", "--output"})
            if (!options.count(name)) throw std::runtime_error(std::string("Missing ") + name);
        const int threads = options.count("--threads") ? std::stoi(options.at("--threads")) : 4;
        const int max_tokens = options.count("--image-max-tokens") ? std::stoi(options.at("--image-max-tokens")) : 256;
        const int reference_tokens = options.count("--reference-max-new") ? std::stoi(options.at("--reference-max-new")) : 0;
        if (threads < 1 || threads > 4 || max_tokens < 64 || max_tokens > 512)
            throw std::runtime_error("Expected 1..4 workers and 64..512 image tokens");
        if (reference_tokens < 0 || reference_tokens > 64) throw std::runtime_error("Reference budget must be 0..64");
        mobile_dflash2::CpuWorkers::pin_owner();
        llama_backend_init();
        auto started = Clock::now();
        auto model_params = llama_model_default_params();
        model_params.vocab_only = reference_tokens == 0;
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(options.at("--text-model").c_str(), model_params), llama_model_free);
        char embedding_width[32]{};
        if (!model || llama_model_meta_val_str(model.get(), "gemma4.embedding_length", embedding_width, sizeof(embedding_width)) < 0 ||
            std::string(embedding_width) != "1536")
            throw std::runtime_error("Expected Gemma4 E2B model metadata with input width 1536");
        // vocab_only intentionally skips loading llama hparams. Validate the
        // projection width from mmproj metadata before reading output vectors.
        std::unique_ptr<gguf_context, decltype(&gguf_free)> projector_info(
            gguf_init_from_file(options.at("--mmproj").c_str(), {true, nullptr}), gguf_free);
        if (!projector_info) throw std::runtime_error("Cannot read projector metadata");
        const auto dim_key = gguf_find_key(projector_info.get(), "clip.vision.projection_dim");
        const auto type_key = gguf_find_key(projector_info.get(), "clip.vision.projector_type");
        if (dim_key < 0 || type_key < 0 || gguf_get_kv_type(projector_info.get(), dim_key) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(projector_info.get(), dim_key) != 1536 ||
            std::string(gguf_get_val_str(projector_info.get(), type_key)) != "gemma4v")
            throw std::runtime_error("Expected Gemma4 E2B vision projector with output width 1536");
        auto params = mtmd_context_params_default();
        params.use_gpu = false;
        params.n_threads = threads;
        params.warmup = false;
        params.image_min_tokens = 64;
        params.image_max_tokens = max_tokens;
        params.print_timings = false;
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> encoder(
            mtmd_init_from_file(options.at("--mmproj").c_str(), model.get(), params), mtmd_free);
        if (!encoder || !mtmd_support_vision(encoder.get())) throw std::runtime_error("Vision mmproj failed to load");
        if (mtmd_decode_use_non_causal(encoder.get(), nullptr) || mtmd_decode_use_mrope(encoder.get()))
            throw std::runtime_error("Only causal, non-MRoPE E2B image inputs are supported");
        const double load_seconds = seconds(started);
        auto image = mtmd_helper_bitmap_init_from_file(encoder.get(), options.at("--image").c_str(), false,
                                                       mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(image.bitmap, mtmd_bitmap_free);
        if (!bitmap || image.video_ctx || mtmd_bitmap_is_audio(bitmap.get()))
            throw std::runtime_error("Expected one static image");
        const auto prompt = read_text(options.at("--prompt-file"));
        mtmd_input_text input{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bitmaps[] = {bitmap.get()};
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        auto preprocessing_start = Clock::now();
        if (mtmd_tokenize(encoder.get(), chunks.get(), &input, bitmaps, 1) != 0)
            throw std::runtime_error("Image tokenization failed; prompt must contain exactly one media marker");
        const double preprocess_seconds = seconds(preprocessing_start);
        std::vector<int64_t> tokens;
        std::vector<float> embeddings;
        json spans = json::array();
        double encode_seconds = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t count = 0;
                auto ids = mtmd_input_chunk_get_tokens_text(chunk, &count);
                tokens.insert(tokens.end(), ids, ids + count);
            } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                auto encoded_start = Clock::now();
                if (mtmd_encode_chunk(encoder.get(), chunk) != 0) throw std::runtime_error("Vision encode failed");
                encode_seconds += seconds(encoded_start);
                const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
                const float* values = mtmd_get_output_embd(encoder.get());
                if (!count || count > 512 || !values) throw std::runtime_error("Invalid vision embedding shape");
                spans.push_back({{"start", tokens.size()}, {"count", count}, {"embedding_row", embeddings.size() / 1536}});
                for (size_t j = 0; j < count * 1536; ++j)
                    if (!std::isfinite(values[j])) throw std::runtime_error("Non-finite projected image embedding");
                embeddings.insert(embeddings.end(), values, values + count * 1536);
                tokens.insert(tokens.end(), count, 258880);
            } else throw std::runtime_error("Audio chunks are not supported");
        }
        if (spans.empty() || tokens.empty() || tokens.size() > 2044)
            throw std::runtime_error("Empty image input or compiled 2K prompt capacity exceeded");
        fs::path output = options.at("--output");
        if (output.has_parent_path()) fs::create_directories(output.parent_path());
        fs::path binary = output;
        binary.replace_extension("embeddings.f32");
        std::ofstream data(binary, std::ios::binary);
        data.write(reinterpret_cast<const char*>(embeddings.data()), embeddings.size() * sizeof(float));
        data.close();
        if (!data) throw std::runtime_error("Image embedding output write failed");
        json result{{"format", "rainfer-vision-v1"}, {"hidden_size", 1536}, {"dtype", "float32-le"},
                    {"tokens", tokens}, {"spans", spans}, {"embeddings_file", binary.filename().string()},
                    {"embedding_rows", embeddings.size() / 1536}, {"image_placeholder_id", 258880},
                    {"embedding_semantics", "projected-unscaled; divide by sqrt(1536) before NPU token-embedding scale"},
                    {"vision_backend", "llama.cpp mtmd CPU"}, {"vision_threads", threads},
                    {"timings", {{"load_seconds", load_seconds}, {"preprocess_seconds", preprocess_seconds},
                                 {"image_encode_seconds", encode_seconds}}}};
        if (reference_tokens) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 2048; cp.n_batch = 256; cp.n_ubatch = 64;
            cp.n_threads = threads; cp.n_threads_batch = threads;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
            if (!context) throw std::runtime_error("GGUF reference context failed");
            llama_pos past = 0;
            auto reference_start = Clock::now();
            if (mtmd_helper_eval_chunks(encoder.get(), context.get(), chunks.get(), 0, 0, 64, true, &past))
                throw std::runtime_error("GGUF vision reference prefill failed");
            const auto vocab = llama_model_get_vocab(model.get());
            std::vector<llama_token> generated;
            std::string decoded;
            for (int i = 0; i < reference_tokens; ++i) {
                const auto logits = llama_get_logits_ith(context.get(), -1);
                if (!logits) throw std::runtime_error("Missing GGUF reference logits");
                llama_token next = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
                generated.push_back(next);
                if (llama_vocab_is_eog(vocab, next)) break;
                std::vector<char> piece(256);
                int size = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, false);
                if (size < 0) { piece.resize(-size); size = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, false); }
                if (size < 0) throw std::runtime_error("Reference detokenization failed");
                decoded.append(piece.data(), size);
                if (i + 1 < reference_tokens) {
                    auto batch = llama_batch_get_one(&next, 1);
                    if (llama_decode(context.get(), batch)) throw std::runtime_error("GGUF reference decode failed");
                }
            }
            result["reference"] = {{"backend", "llama.cpp Q4 GGUF CPU; no QNN or DFlash2"},
                                   {"tokens", generated}, {"text", decoded}, {"total_seconds", seconds(reference_start)}};
        }
        std::ofstream manifest(output);
        manifest << result.dump(2) << '\n';
        manifest.close();
        if (!manifest) throw std::runtime_error("Vision bundle manifest write failed");
        const auto checked = VisionBundle::load(output.string());
        if (checked.images->image_rows() != embeddings.size() / 1536)
            throw std::runtime_error("Bundle round-trip changed image rows");
        std::cout << json{{"status", "encoded"}, {"prompt_tokens", tokens.size()},
                          {"image_tokens", embeddings.size() / 1536}, {"image_encode_seconds", encode_seconds},
                          {"bundle_round_trip_verified", true}, {"reference", result.value("reference", json())}}.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
