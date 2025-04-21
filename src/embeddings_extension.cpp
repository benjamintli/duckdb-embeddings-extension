#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "embeddings_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void EmbeddingsScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	auto count = args.size();
	for (idx_t i = 0; i < count; i++) {
		vector<Value> embedding;

		for (int i = 0; i < 10; i++) {
			embedding.push_back(1.0);
		}
		result.SetValue(i, Value::LIST(embedding));
	}
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a scalar function
	auto embedding_scalar_functions = ScalarFunction("embeddings", {LogicalType::VARCHAR},
	                                                 LogicalType::LIST(LogicalType::FLOAT), EmbeddingsScalarFun);
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
