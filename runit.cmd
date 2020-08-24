@echo off
cargo run input.txt output.exe
output.exe
echo errorlevel = %errorlevel%
