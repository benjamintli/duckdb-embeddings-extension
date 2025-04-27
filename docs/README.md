# DuckDB EmbedText: Generating Text Embeddings directly in SQL
EmbedText is an ML extension for DuckDB that enables you to easily generate text embeddings on your data with a scalar function. It uses Hugging Face models for embedding generation, and runs locally with Candle under the hood for optimal performance. No need to rely on external services — run it all on your machine! 🖥️💡

## Usage
```
INSTALL embed_text FROM https://github.com/benjamintli/duckdb-textembed-extension;
LOAD embed_text

SELECT embed_text('<model name>', <column_name>) from <table_name>
```

## Motivation
DuckDB (at least to me) is one of those "batteries included" DB tools that let you do analysis on data easily. Part of the motivation behind why people use DuckDB is that it's a lot easier to do analysis straight in SQL than it is to use something like pandas. The "batteries included" also lets people skip writing repetitive ETL glue. There's a lot of stuff in DuckDB that makes for a good developer experience, like being able to easily import CSV files, parquet, talk to S3, and more recently, being able to do things like embedding search and similarity. 

However, if your dataset you're importing doesn't already contain some embeddings, you're forced to write some Python glue code to iterate through rows and create embeddings, thus defeating the purpose of using DuckDB. Here's where EmbedText comes in; you can create embeddings directly in SQL, allowing you to avoid writing that piece of ETL glue to create the embeddings.
