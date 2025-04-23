#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "text_embedding/lib.h"
#include <memory>

namespace duckdb {

struct EmbeddingFunctionData : public FunctionData {
	std::shared_ptr<TextEmbedder> text_embedder_;

	explicit EmbeddingFunctionData(std::shared_ptr<TextEmbedder> text_embedder)
	    : text_embedder_(std::move(text_embedder)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<EmbeddingFunctionData>(text_embedder_);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<EmbeddingFunctionData>();
		return text_embedder_ == other.text_embedder_;
	}
};

struct EmbeddingsLocalState : public FunctionLocalState {
	explicit EmbeddingsLocalState(std::shared_ptr<TextEmbedder> text_embedder)
	    : text_embedder_ {std::move(text_embedder)} {
	}

	void embed(const std::string &text, std::vector<Value> &embeddings_output) {
		embeddings_output.reserve(text_embedder_->get_output_dims());
		for (const auto &embedding : text_embedder_->embed(text)) {
			embeddings_output.emplace_back(embedding);
		}
	}

private:
	std::shared_ptr<TextEmbedder> text_embedder_;
};

} // namespace duckdb