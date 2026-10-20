set positional-arguments

check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

presets:
    cargo run -p burns --no-default-features --example generate_presets

cli *ARGS:
    cargo run --release -p burns-cli -- "$@"

gui *ARGS:
    cargo run --release -p burns-gui -- "$@"

wasm:
    cargo build --release -p burns-wasm --target wasm32-unknown-unknown
    wasm-bindgen --target web --out-dir site/src/wasm --out-name burns target/wasm32-unknown-unknown/release/burns_wasm.wasm

site: wasm
    pnpm --dir site install --frozen-lockfile
    pnpm --dir site build

site-serve PORT="8472": wasm
    pnpm --dir site dev --host 127.0.0.1 --port "$1" --strictPort
