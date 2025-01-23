@echo off
setlocal
set RUST_LOG=lex=info,parse=debug,lrncc=info
cargo run -- -v input.c output.exe
output.exe
echo errorlevel = %errorlevel%
