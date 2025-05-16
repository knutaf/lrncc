setlocal
if not defined RUST_LOG set RUST_LOG=lex=info,parse=debug,resolve=debug,consteval=debug,lrncc=debug
cargo test -- %*
