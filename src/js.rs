use js_sys::Reflect;
use serde::{Deserialize, Serialize};
use std::io;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    static performance: web_sys::Performance;

    fn postMessage(value: &JsValue);
}

pub fn now() -> f64 {
    performance.now()
}

#[derive(Deserialize)]
#[serde(tag = "cmd")]
pub enum Request {
    RunSolve {
        words: String,
        guesses: String,
        num_roots: u32,
        dk_trunc: f64,
    },
}

#[derive(Serialize)]
#[serde(tag = "cmd")]
pub enum Reply {
    UpdateStatus {
        message: String,
        progress: f64,
    },
    ReportStrategy {
        depths: Vec<usize>,
        depth_avg: f64,
        decision_tree: crate::Decision,
    },
    SetCandidates {
        candidates: Vec<String>,
    },
}

impl Reply {
    pub fn post(&self) {
        postMessage(&JsValue::from_serde(self).unwrap());
    }
}

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn onmessage(e: &JsValue) {
    if let Err(err) = handle_message(&Reflect::get(e, &"data".into()).unwrap()) {
        update_status(format!("Error: {}", err));
    }
}

fn handle_message(data: &JsValue) -> io::Result<()> {
    match data.into_serde()? {
        Request::RunSolve {
            words,
            guesses,
            num_roots,
            dk_trunc,
        } => crate::solve(&words, &guesses, num_roots, dk_trunc),
    }
}

pub fn log(message: &str) {
    console::log_1(&message.into());
}

pub fn update_status(message: String) {
    update_progress(message, f64::NAN);
}

pub fn update_progress(message: String, progress: f64) {
    Reply::UpdateStatus { message, progress }.post();
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
