use anyhow::Error;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

const MAX_BATCH_SIZE: usize = 64;

pub enum BertModelTypes {
    SnowflakeArcticEmbedL,
    UaeLargeV1,
    MxbaiEmbedLargeV1,
    SnowflakeArcticEmbedM,
    BgeLargeEnV15,
    BgeBaseEnV15,
    AllMinilmL6V2,
}

impl BertModelTypes {
    pub fn to_hf_hub_id(&self) -> &str {
        match self {
            BertModelTypes::SnowflakeArcticEmbedL => "Snowflake/snowflake-arctic-embed-l",
            BertModelTypes::UaeLargeV1 => "WhereIsAI/UAE-Large-V1",
            BertModelTypes::MxbaiEmbedLargeV1 => "mixedbread-ai/mxbai-embed-large-v1",
            BertModelTypes::SnowflakeArcticEmbedM => "Snowflake/snowflake-arctic-embed-m",
            BertModelTypes::BgeLargeEnV15 => "BAAI/bge-large-en-v1.5",
            BertModelTypes::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            BertModelTypes::AllMinilmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, Error> {
        // trim the sql string prefix/suffix
        let s = s.trim_matches('\'');
        match s {
            "Snowflake/snowflake-arctic-embed-l" => Ok(BertModelTypes::SnowflakeArcticEmbedL),
            "WhereIsAI/UAE-Large-V1" => Ok(BertModelTypes::UaeLargeV1),
            "mixedbread-ai/mxbai-embed-large-v1" => Ok(BertModelTypes::MxbaiEmbedLargeV1),
            "Snowflake/snowflake-arctic-embed-m" => Ok(BertModelTypes::SnowflakeArcticEmbedM),
            "BAAI/bge-large-en-v1.5" => Ok(BertModelTypes::BgeLargeEnV15),
            "BAAI/bge-base-en-v1.5" => Ok(BertModelTypes::BgeBaseEnV15),
            "sentence-transformers/all-MiniLM-L6-v2" => Ok(BertModelTypes::AllMinilmL6V2),
            _ => Err(anyhow::Error::msg(format!("{} is not a known bert model!", s))),
        }
    }

    pub fn all_hf_hub_ids_as_string() -> String {
        let all_models = [
            BertModelTypes::SnowflakeArcticEmbedL,
            BertModelTypes::UaeLargeV1,
            BertModelTypes::MxbaiEmbedLargeV1,
            BertModelTypes::SnowflakeArcticEmbedM,
            BertModelTypes::BgeLargeEnV15,
            BertModelTypes::BgeBaseEnV15,
            BertModelTypes::AllMinilmL6V2,
        ];

        all_models
            .iter()
            .map(|model| model.to_hf_hub_id())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub struct TextEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
}

fn get_device() -> Result<Device, Error> {
    #[cfg(target_os = "macos")]
    return Ok(Device::new_metal(0)?);
    #[cfg(not(target_os = "macos"))]
    return Ok(Device::Cpu);
}

impl TextEmbedder {
    pub fn new(model_id: &str) -> Result<Self, Error> {
        let device = get_device()?;
        let (model, tokenizer, config) = build_model_and_tokenizer(model_id, &device)?;
        Ok(TextEmbedder {
            model,
            tokenizer,
            device,
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

    pub fn embed_batch(&self, prompts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if prompts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::new();

        for batch_prompts in prompts.chunks(MAX_BATCH_SIZE) {
            // Tokenize the batch
            let tokens = self
                .tokenizer
                .encode_batch(batch_prompts.to_vec(), true)
                .map_err(Error::msg)?;

            let token_ids = tokens
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_ids().to_vec();
                    Ok(Tensor::new(tokens.as_slice(), &self.device)?)
                })
                .collect::<Result<Vec<_>, Error>>()?;

            let token_ids = Tensor::stack(&token_ids, 0)?;
            let token_type_ids = token_ids.zeros_like()?;

            let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
            let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
            let embeddings = self.normalize_l2(&embeddings)?;
            let embeddings = embeddings.squeeze(1)?;
            let batch_embeddings = embeddings.to_vec2()?;

            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    pub fn get_output_dims(&self) -> usize {
        return self.config.hidden_size;
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor, Error> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}

fn build_model_and_tokenizer(
    model_id: &str,
    device: &Device,
) -> Result<(BertModel, Tokenizer, Config), Error> {
    let model = BertModelTypes::from_str(model_id)?;

    let repo = Repo::model(BertModelTypes::to_hf_hub_id(&model).to_string());
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
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check for TextEmbedder::new() + embed().
    ///
    /// This is a slow, network-dependent test, so it’s ignored by default.
    #[ignore]
    #[test]
    fn sanity_embedder_download_and_embed() {
        // 1. Construct (which downloads & builds the model)
        let te = TextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")
            .expect("TextEmbedder::new() failed");

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

    #[ignore]
    #[test]
    fn sanity_embedder_batch_embed() {
        // 1. Construct (downloads & builds the model)
        let te = TextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")
            .expect("TextEmbedder::new() failed");

        // 2. Batch embed some prompts
        let prompts = &[
            "the quick brown fox",
            "jumps over the lazy dog",
            "and runs away",
        ];
        let embeddings = te
            .embed_batch(prompts)
            .expect("embed_batch() returned an error");

        // 3. Check shape
        assert_eq!(
            embeddings.len(),
            prompts.len(),
            "Got {} embeddings but {} prompts",
            embeddings.len(),
            prompts.len()
        );

        let expected_dim = te.get_output_dims();
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                expected_dim,
                "Embedding {} has length {}, expected {}",
                i,
                embedding.len(),
                expected_dim
            );

            // 4. Check normalization (L2 norm ≈ 1.0)
            let norm_sq: f32 = embedding.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            let eps = 1e-3;
            assert!(
                (norm - 1.0).abs() < eps,
                "Embedding {} L2 norm = {}, but expected ~1.0",
                i,
                norm
            );
        }
    }
}
