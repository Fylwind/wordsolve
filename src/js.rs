use crate::Log;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

#[wasm_bindgen]
extern "C" {
    fn logAppend(s: &str);

    pub fn now() -> f64;

    pub fn postMessage(value: &JsValue);
}

#[derive(Serialize)]
#[serde(tag = "cmd")]
pub enum OutMessage {
    ClearLog,
    Log { message: String },
    SetCandidates { candidates: Vec<String> },
}

impl OutMessage {
    pub fn post(&self) {
        postMessage(&JsValue::from_serde(self).unwrap());
    }
}

#[derive(Deserialize)]
#[serde(tag = "cmd")]
pub enum InMessage {
    RunSolve { words: Vec<String>, guesses: String },
}

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn onmessage(value: &JsValue) {
    if let Err(err) = match value.into_serde().unwrap() {
        InMessage::RunSolve { words, guesses } => crate::solve(&words, &guesses, &JsLog),
    } {
        JsLog.log(&format!("Error: {}", err));
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JsLog;

impl Log for JsLog {
    fn log(&self, message: &str) {
        OutMessage::Log {
            message: message.into(),
        }
        .post();
    }
}
