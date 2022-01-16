"use strict";

importScripts('./pkg/wordsolve.js');
const lib = wasm_bindgen;
const initLib = lib('./pkg/wordsolve_bg.wasm');

async function main() {
    await initLib;
    lib.init();
    onmessage = lib.onmessage;
}

main();
