use anyhow::Error;
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

pub struct TextEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
}

fn build_model_and_tokenizer(model_id: &str) -> Result<(BertModel, Tokenizer, Config), Error> {
    let device = Device::Cpu;
    let default_model = model_id.trim_matches('\'').to_string();

    let repo = Repo::model(default_model);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer, config))
}

impl TextEmbedder {
    pub fn new(model_id: &str) -> Result<Self, Error> {
        let (model, tokenizer, config) = build_model_and_tokenizer(model_id)?;
        Ok(TextEmbedder {
            model,
            tokenizer,
            device: Device::Cpu,
            config,
        })
    }

    pub fn embed(&self, prompt: &str) -> Result<Vec<f32>, Error> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = self.normalize_l2(&embeddings)?;
        return Ok(embeddings.squeeze(0)?.to_vec1()?);
    }

    pub fn get_output_dims(&self) -> usize {
        return self.config.hidden_size;
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor, Error> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check for TextEmbedder::new() + embed().
    ///
    /// This is a slow, network-dependent test, so it’s ignored by default.
    #[test]
    fn sanity_embedder_download_and_embed() {
        // 1. Construct (which downloads & builds the model)
        let te = TextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2").expect("TextEmbedder::new() failed");

        // 2. Run embed()
        let prompt = "the quick brown fox jumps over the lazy dog";
        let embedding = te.embed(prompt).expect("embed() returned an error");

        // 3. Check dimension
        println!("{:?}", embedding);
        let expected_dim = te.get_output_dims();
        assert_eq!(
            embedding.len(),
            expected_dim,
            "Embedding length {}, expected {}",
            embedding.len(),
            expected_dim
        );

        // 4. Check normalization (L2 norm ≈ 1.0)
        let norm_sq: f32 = embedding.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        let eps = 1e-3;
        assert!(
            (norm - 1.0).abs() < eps,
            "L2 norm = {}, but expected ~1.0",
            norm
        );
    }
}
