use crate::Log;
use js_sys::Reflect;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    static performance: web_sys::Performance;

    fn logAppend(s: &str);

    fn postMessage(value: &JsValue);
}

pub fn now() -> f64 {
    performance.now()
}

#[derive(Serialize)]
#[serde(tag = "cmd")]
pub enum OutMessage {
    UpdateStatus { message: String, progress: f64 },
    AppendQuery { query: Vec<String> },
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
    RunSolve {
        words: String,
        guesses: String,
        max_branching: f64,
    },
}

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn onmessage(e: &JsValue) {
    let data = Reflect::get(e, &"data".into()).unwrap();
    if let Err(err) = match data.into_serde().unwrap() {
        InMessage::RunSolve {
            words,
            guesses,
            max_branching,
        } => crate::solve(&words, &guesses, max_branching as _, &JsLog),
    } {
        update_status(format!("Error: {}", err));
    }
}

pub fn update_status(message: String) {
    update_progress(message, f64::NAN);
}

pub fn update_progress(message: String, progress: f64) {
    OutMessage::UpdateStatus { message, progress }.post();
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JsLog;

impl Log for JsLog {
    fn log(&self, message: &str) {
        console::log_1(&message.into());
    }
}

#[derive(Debug)]
pub struct Timer {
    name: &'static str,
}

impl From<&'static str> for Timer {
    fn from(name: &'static str) -> Self {
        console::time_with_label(name);
        Self { name }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        console::time_end_with_label(self.name);
    }
}
