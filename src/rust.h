#include <cstddef>
#include <cstdlib>


struct TextEmbedder;

/// Opaque handle
using TextEmbedderHandle = TextEmbedder*;


extern "C" {

/// Create a new TextEmbedder and return a raw pointer
TextEmbedderHandle text_embedder_create(const char *model_id);

/// Embed; returns a newly-malloc’d float array and writes its length
float *text_embedder_embed(TextEmbedderHandle h, const char *prompt, size_t *out_len);

/// Drop it
void text_embedder_free(TextEmbedderHandle h);

/// Free the float array returned by `embed`
void text_embedder_free_f32(float *ptr, size_t len);

/// Get the output dimension
size_t text_embedder_output_dims(TextEmbedderHandle h);

}  // extern "C"
