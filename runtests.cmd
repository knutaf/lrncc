setlocal
if not defined RUST_LOG set RUST_LOG=lex=info,parse=debug,lrncc=debug
cargo test -- %*
