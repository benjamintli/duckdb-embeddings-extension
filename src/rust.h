#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>


struct TextEmbedder;

/// Opaque handle
using TextEmbedderHandle = TextEmbedder*;


extern "C" {

/// Create a new TextEmbedder and return a raw pointer
TextEmbedderHandle text_embedder_create(const char *model_id, char **out_error);

/// Embed; returns a newly-malloc’d float array and writes its length
float *text_embedder_embed(TextEmbedderHandle h, const char *prompt, size_t *out_len);

float *text_embedder_embed_batch(TextEmbedderHandle h,
                                 const char *const *prompts,
                                 size_t n_prompts,
                                 size_t *out_total_len);

/// Drop it
void text_embedder_free(TextEmbedderHandle h);

/// Free the float array returned by `embed`
void text_embedder_free_f32(float *ptr, size_t len);

void text_embedder_free_string(char *str);

char *text_embedder_list_models();

/// Get the output dimension
size_t text_embedder_output_dims(TextEmbedderHandle h);

}  // extern "C"
