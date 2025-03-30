pub struct ModelHandler {
    model_name: String, // list of downloaded models
    models_dir: String, // path to the models directory
}

const MODEL_MAP: phf::Map<&'static str, &'static str> = phf::phf_map! {
    "tiny" => "ggml-tiny",
    "base" => "ggml-base",
    "small" => "ggml-small",
    "medium" => "ggml-medium",
    "large" => "ggml-large-v3",
};

impl ModelHandler {
    pub async fn new(model_name: &str, models_dir: &str) -> ModelHandler {
        let model_handler = ModelHandler {
            model_name: MODEL_MAP
                .get(&model_name.trim().to_lowercase())
                .ok_or_else(|| format!("Model 'ggml-{}' not found", model_name))
                .copied()
                .unwrap()
                .to_string(),
            models_dir: models_dir.to_string(),
        };

        if model_handler.is_model_existing() {
            return model_handler;
        }

        let _ = model_handler.setup_directory();
        let _ = model_handler.download_model().await;

        model_handler
    }

    /// setup the directory to which models will be downloaded.
    /// Sets a global vx
    ///
    /// # Returns
    ///
    /// * `Void` - directory is setup.
    fn setup_directory(&self) -> Result<(), std::io::Error> {
        let path = std::path::Path::new(&self.models_dir);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }

    fn is_model_existing(&self) -> bool {
        std::fs::metadata(format!("{}/{}.bin", self.models_dir, self.model_name)).is_ok()
    }

    /// Download the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to download.
    ///
    /// # Returns
    ///
    /// * `Void` - The model is downloaded to the models directory.
    async fn download_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_model_existing() {
            self.setup_directory()?;
        }
        let base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
        let model_file_url = format!("{}/{}.bin", base_url, &self.model_name);
        println!("Downloading model from: {}", model_file_url);
        let response = reqwest::get(model_file_url).await?;
        let mut file = std::fs::File::create(self.get_model_dir())?;
        let mut content = std::io::Cursor::new(response.bytes().await?);
        std::io::copy(&mut content, &mut file)?;
        Ok(())
    }

    pub fn get_model_dir(&self) -> String {
        format!("{}/{}.bin", &self.models_dir, &self.model_name)
    }
}

#[cfg(test)]
mod tests {
    use crate::model_handler;

    const MODEL_DIR: &'static str = "test_models/";

    #[tokio::test]
    async fn test_check_model_exists_existent_path() {
        let path = std::path::Path::new(&format!("{}/ggml-tiny.bin", MODEL_DIR));
        if !path.exists() {
            let _ = std::fs::create_dir_all(path);
        }

        let test_model = model_handler::ModelHandler::new("tiny", MODEL_DIR).await;
        let result = test_model.is_model_existing();
        assert_eq!(result, true);
        let _ = std::fs::remove_dir_all(MODEL_DIR);
    }

    #[tokio::test]
    async fn test_setup_directory_happy_case() {
        let path = std::path::Path::new(&format!("{}/ggml-tiny.bin", MODEL_DIR));
        if !path.exists() {
            let _ = std::fs::create_dir_all(path);
        }

        let test_model = model_handler::ModelHandler::new("tiny", MODEL_DIR).await;
        let result = test_model.setup_directory();
        assert_eq!(result.is_ok(), true);
        let _ = std::fs::remove_dir_all(MODEL_DIR);
    }

    #[tokio::test]
    async fn test_download_model_happy_case() {
        const MODEL_DIR: &'static str = "test_models/";

        fn prep_test_dir() {
            let path = std::path::Path::new(MODEL_DIR);
            if !path.exists() {
                let _ = std::fs::create_dir_all(path);
            }
        }

        prep_test_dir();

        let model_handler = model_handler::ModelHandler::new("tiny", MODEL_DIR).await;

        let _result = model_handler.download_model().await;

        let is_file_existing = match std::fs::metadata(format!("{}/ggml-tiny.bin", MODEL_DIR)) {
            Ok(_) => true,
            Err(_) => false,
        };

        assert_eq!(is_file_existing, true);

        let _ = std::fs::remove_dir_all(MODEL_DIR);
    }
}
