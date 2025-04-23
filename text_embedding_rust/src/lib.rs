
pub struct TextEmbedder {
    model: String,
    output_dims: u32,
}

impl Default for TextEmbedder {
    fn default() -> Self {
        Self {
            model: String::new(),
            output_dims: 0,
        }
    }
}

impl TextEmbedder {
    pub fn embed(&self, text: &str) -> Vec<f32> {
        println!("Embedding {}", text);
        return vec![0.0, 1.0, 2.0];
    }

    pub fn init(&self, model_name: &str) -> Result<(), anyhow::Error> {
        println!("initializing model: {}", model_name);
        if model_name.is_empty() {
            return Err(anyhow::Error::msg("error thrown!"));
        }
        Ok(())
    }

    pub fn get_model_name(&self) -> &str {
        return &self.model;
    }

    pub fn get_output_dims(&self) -> u32 {
        return self.output_dims;
    }
}

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type TextEmbedder;
        fn init(self: &TextEmbedder, model_name: &str) -> Result<()>;
        fn embed(self: &TextEmbedder, text: &str) -> Vec<f32>;
        fn get_model_name(self: &TextEmbedder) -> &str;
        fn get_output_dims(self: &TextEmbedder) -> u32;
    }
}
