use whisper_stt::model_handler::ModelHandler;
use whisper_stt::transcriber::Transcriber;

#[tokio::main]
async fn main() {
    let m = ModelHandler::new("tiny", "models/").await;
    let trans = Transcriber::new(m);
    let result = trans.transcribe("src/test_data/test.mp3", None).unwrap();
    let text = result.get_text();
    let start = result.get_start_timestamp();
    let end = result.get_end_timestamp();
    println!("start[{}]-end[{}] {}", start, end, text);
}
