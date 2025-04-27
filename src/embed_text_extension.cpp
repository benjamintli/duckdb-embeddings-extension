#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/function.hpp"
#include "text_embedder.hpp"
#include <memory>
#include <stdexcept>
#define DUCKDB_EXTENSION_MAIN

#include "embed_text_extension.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

namespace duckdb {
// NOLINTNEXTLINE: singleton
std::unordered_map<std::string, std::shared_ptr<TextEmbedder>> TextEmbedder::instances;

inline void EmbedTextBatchScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &model_name = args.data[0];
	auto &column_name = args.data[1];
	auto model_string_ptr = ConstantVector::GetData<string_t>(model_name);
	auto row_counts = args.size();
	std::vector<std::string> inputs;
	inputs.reserve(row_counts);
	for (idx_t i = 0; i < row_counts; i++) {
		inputs.push_back(column_name.GetValue(i).ToString());
	}
	const auto &embeddings = TextEmbedder::getInstance(model_string_ptr->GetString())->embed_batch(inputs);
	for (idx_t i = 0; i < row_counts; i++) {
		result.SetValue(i, Value::LIST(LogicalTypeId::FLOAT, embeddings[i]));
	}
}

static unique_ptr<FunctionData> EmbedTextBindingFun(ClientContext &context, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidTypeException("model name is not a constant string!");
	}

	// initialize the model on bind
	TextEmbedder::getInstance(arguments[0]->ToString());
	return nullptr;
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a scalar function
	auto embedding_scalar_functions =
	    ScalarFunction("embed_text", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                   LogicalType::LIST(LogicalType::FLOAT), EmbedTextBatchScalarFun, EmbedTextBindingFun);
	ExtensionUtil::RegisterFunction(instance, embedding_scalar_functions);
}

void EmbedTextExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string EmbedTextExtension::Name() {
	return "embed_text";
}

std::string EmbedTextExtension::Version() const {
#ifdef EXT_VERSION_QUACK
	return EXT_VERSION_QUACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void embed_text_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::EmbedTextExtension>();
}

DUCKDB_EXTENSION_API const char *embed_text_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
