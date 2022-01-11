set -eux
cargo check --examples
wasm-pack build --release --target no-modules
