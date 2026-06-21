@echo off
REM ================================================================
REM  RETRAIN WITH EXPORT - Wrapper script
REM  
REM  1. Exports new claims from MySQL → data/new/claims_YYYYMMDD.csv
REM  2. Runs retrain_cycle.py with the new data
REM
REM  Usage (manual):
REM      retrain_with_export.bat
REM      retrain_with_export.bat --since 7d
REM      retrain_with_export.bat --since 2026-06-01
REM
REM  For Windows Task Scheduler:
REM      Program: E:\SCHOOL\PTL\retrain_with_export.bat
REM      Start in: E:\SCHOOL\PTL
REM ================================================================

setlocal enabledelayedexpansion

REM --- Configuration ---
set "PROJECT_DIR=E:\SCHOOL\PTL"
set "PYTHON=python"
set "SINCE=%~1"
set "RETRAIN_DAYS=4"

cd /d "%PROJECT_DIR%"

echo ================================================================
echo  RETRAIN WITH EXPORT PIPELINE
echo  Started: %date% %time%
echo ================================================================
echo.

REM --- Step 1: Export claims from database ---
echo [STEP 1/2] Exporting claims from MySQL...
echo.

if "%SINCE%"=="" (
    echo   Using default: export ALL claims
    %PYTHON% src/export_claims_csv.py
) else (
    echo   Using --since %SINCE%
    %PYTHON% src/export_claims_csv.py --since %SINCE%
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Export failed with error code %ERRORLEVEL%.
    echo         Retrain will still proceed with existing data if available.
    echo.
)

echo.

REM --- Step 2: Run retrain cycle ---
echo [STEP 2/2] Running retrain cycle...
echo.

%PYTHON% src/retrain_cycle.py --new-data-dir data/new --days %RETRAIN_DAYS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Retrain cycle failed with error code %ERRORLEVEL%.
    echo ================================================================
    exit /b 1
)

echo.
echo ================================================================
echo  RETRAIN PIPELINE COMPLETED SUCCESSFULLY
echo  Finished: %date% %time%
echo ================================================================

endlocal
exit /b 0
