@echo off
setlocal
if not defined RUST_LOG set RUST_LOG=lex=info,parse=debug,resolve=info,consteval=info,lrncc=info,asm=debug
cargo run -- %* -v input.c output.exe
output.exe
echo errorlevel = %errorlevel%
