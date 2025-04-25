#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"
#include "text_embedder.hpp"
#include <memory>
#include <mutex>
#include <vector>

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

	std::vector<Value> embed(const std::string &text) {
		return text_embedder_->embed(text);
	}

private:
	std::shared_ptr<TextEmbedder> text_embedder_;
};

} // namespace duckdb