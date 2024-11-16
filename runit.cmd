@echo off
cargo run input.c output.exe
output.exe
echo errorlevel = %errorlevel%
