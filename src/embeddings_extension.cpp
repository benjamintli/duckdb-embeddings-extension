#include "duckdb/common/helper.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/function.hpp"
#include "function_data.hpp"
#include "text_embedding/lib.h"
#include "rust/cxx.h"
#define DUCKDB_EXTENSION_MAIN

#include "embeddings_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

namespace duckdb {

inline void EmbeddingsScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_name = args.data[0];
	auto &column_name = args.data[1];
	auto &lstate = state.Cast<EmbeddingsLocalState>();
	auto count = args.size();
	for (idx_t i = 0; i < count; i++) {
		vector<Value> embedding;
		lstate.embed(column_name.GetValue(i).ToString(), embedding);
		result.SetValue(i, Value::LIST(embedding));
	}
}

static unique_ptr<FunctionData> EmbeddingsBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	auto model_name = arguments[0]->ToString();
	auto text_embedder = std::shared_ptr<TextEmbedder>();
	try {
		text_embedder->init(model_name);
	} catch (const rust::Error &e) {
		throw InvalidInputException(e.what());
	}
	return make_uniq<EmbeddingFunctionData>(std::move(text_embedder));
}

static unique_ptr<FunctionLocalState>
EmbeddingsInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr, FunctionData *bind_data) {
	auto &embeddings_function_data = bind_data->Cast<EmbeddingFunctionData>();
	return make_uniq<EmbeddingsLocalState>(embeddings_function_data.text_embedder_);
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a scalar function
	auto embedding_scalar_functions = ScalarFunction(
	    "embeddings", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::LIST(LogicalType::FLOAT),
	    EmbeddingsScalarFun, EmbeddingsBindFunction, nullptr, nullptr, EmbeddingsInitLocalState);
	ExtensionUtil::RegisterFunction(instance, embedding_scalar_functions);
}

void EmbeddingsExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string EmbeddingsExtension::Name() {
	return "embeddings";
}

std::string EmbeddingsExtension::Version() const {
#ifdef EXT_VERSION_QUACK
	return EXT_VERSION_QUACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void embeddings_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::EmbeddingsExtension>();
}

DUCKDB_EXTENSION_API const char *embeddings_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
