# DuckDB TextEmbed: Generating Text Embeddings directly in SQL
TextEmbed is an ML extension for DuckDB that enables you to easily generate text embeddings on your data with a scalar function. It uses Hugging Face models for embedding generation, and runs locally with Candle under the hood for optimal performance. No need to rely on external services — run it all on your machine! 🖥️💡

## Motivation
DuckDB is a useful tool for data folks who want to do quick analysis without having to q

## Usage
```
INSTALL embed_text FROM https://github.com/benjamintli/duckdb-textembed-extension;
LOAD embed_text
```