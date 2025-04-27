// TextEmbedder.hpp
#pragma once

#include "rust.h"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

namespace {
constexpr static size_t BATCH_SIZE = 64;
}

namespace duckdb {

class TextEmbedder {
public:
	static std::shared_ptr<TextEmbedder> getInstance(const std::string &model_id) {
		// Check if we already have an instance for this model path
		auto it = instances.find(model_id);
		if (it != instances.end()) {
			return it->second;
		}

		// Create new instance and store in the registry
		auto instance = std::shared_ptr<TextEmbedder>(new TextEmbedder(model_id));
		instances[model_id] = instance;
		return instance;
	}

	/// Constructs the embedder (calls text_embedder_create)
	explicit TextEmbedder(const std::string &model_id) {
		handle_ = text_embedder_create(model_id.c_str());
		if (!handle_) {
			throw std::runtime_error("Failed to create TextEmbedder");
		}
	}

	/// Runs embedding on the given prompt and returns a std::vector<duckdb::Value>
	std::vector<duckdb::Value> embed(const std::string &prompt) {
		size_t len = 0;
		float *raw = text_embedder_embed(handle_, prompt.c_str(), &len);
		if (!raw) {
			throw std::runtime_error("TextEmbedder::embed failed");
		}
		// copy into std::vector and free the C-allocated buffer
		std::vector<duckdb::Value> out(raw, raw + len);
		text_embedder_free_f32(raw, len);
		return out;
	}

	vector<vector<duckdb::Value>> embed_batch(const std::vector<const char *> &prompts) {
		// Create pointers to store the inputs
		size_t out_lengths = 0;
		float *raw_embeddings = text_embedder_embed_batch(handle_, &prompts[0], prompts.size(), &out_lengths);
		vector<vector<duckdb::Value>> embeddings;
		embeddings.reserve(prompts.size());
		for (size_t i = 0; i < out_lengths; i += output_dims()) {
			vector<duckdb::Value> embedding;
			embedding.reserve(output_dims());
			for (size_t j = i; j < output_dims() + i; j++) {
				embedding.emplace_back(raw_embeddings[j]);
			}
			embeddings.push_back(embedding);
		}
		text_embedder_free_f32(raw_embeddings, out_lengths);
		return embeddings;
	}

	/// Returns the embedding dimensionality
	size_t output_dims() const {
		return text_embedder_output_dims(handle_);
	}

	/// Clean up (calls text_embedder_free)
	~TextEmbedder() {
		if (handle_) {
			text_embedder_free(handle_);
			handle_ = nullptr;
		}
	}

	// non-copyable
	TextEmbedder(const TextEmbedder &) = delete;
	TextEmbedder &operator=(const TextEmbedder &) = delete;

	// movable
	TextEmbedder(TextEmbedder &&other) noexcept : handle_(other.handle_) {
		other.handle_ = nullptr;
	}
	TextEmbedder &operator=(TextEmbedder &&other) noexcept {
		if (this != &other) {
			if (handle_) {
				text_embedder_free(handle_);
			}
			handle_ = other.handle_;
			other.handle_ = nullptr;
		}
		return *this;
	}

private:
	TextEmbedderHandle handle_ {nullptr};
	// NOLINTNEXTLINE: singleton
	static std::unordered_map<std::string, std::shared_ptr<TextEmbedder>> instances;
};

} // namespace duckdb