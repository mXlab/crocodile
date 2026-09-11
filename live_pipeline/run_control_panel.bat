@echo off
setlocal

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

open-stage-control ^
    --load "%REPO_ROOT%\live_pipeline\crocodile-control-panel.json" ^
    --port 8090 ^
    --send 127.0.0.1:9000 ^
    --osc-port 9001 ^
    --custom-module "%REPO_ROOT%\live_pipeline\crocodile-control-module.js" ^
    %*

endlocal