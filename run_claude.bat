@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM CONFIG
REM =====================================================

set PROMPT_DIR=prompts
set LOG_DIR=logs

REM =====================================================
REM CREATE LOG DIR
REM =====================================================

if not exist %LOG_DIR% (
    mkdir %LOG_DIR%
)

REM =====================================================
REM PROCESS ALL PROMPTS
REM =====================================================

for %%f in (%PROMPT_DIR%\*.txt) do (

    echo ============================================
    echo RUNNING: %%~nxf
    echo ============================================

    set LOGFILE=%LOG_DIR%\%%~nf.log

    echo Prompt: %%f > !LOGFILE!
    echo -------------------------------------------- >> !LOGFILE!

    type "%%f" | claude >> !LOGFILE! 2>&1

    echo.
    echo Finished %%~nxf
    echo Log saved to !LOGFILE!
    echo.
)

echo ALL DONE
pause