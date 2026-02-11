@echo off
REM Model Retraining Scheduler Management Script (Windows)
REM This script provides easy commands to manage the model retraining scheduler

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PYTHON_CMD=python"
set "SCHEDULER_SCRIPT=%SCRIPT_DIR%retraining_scheduler.py"

REM Colors for output (Windows CMD)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

REM Function to print colored output
:print_status
echo [%GREEN%INFO%RESET%] %~1
goto :eof

:print_warning
echo [%YELLOW%WARNING%RESET%] %~1
goto :eof

:print_error
echo [%RED%ERROR%RESET%] %~1
goto :eof

:print_info
echo [%BLUE%INFO%RESET%] %~1
goto :eof

REM Check if Python is available
:check_python
where "%PYTHON_CMD%" >nul 2>nul
if %errorlevel% neq 0 (
    call :print_error "Python command '%PYTHON_CMD%' not found"
    exit /b 1
)
goto :eof

REM Check if scheduler script exists
:check_scheduler_script
if not exist "%SCHEDULER_SCRIPT%" (
    call :print_error "Scheduler script not found: %SCHEDULER_SCRIPT%"
    exit /b 1
)
goto :eof

REM Start the scheduler
:start_scheduler
call :print_info "Starting Model Retraining Scheduler..."
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --start
goto :eof

REM Stop the scheduler
:stop_scheduler
call :print_info "Stopping Model Retraining Scheduler..."
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --stop
goto :eof

REM Check scheduler status
:status_scheduler
call :print_info "Checking scheduler status..."
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --status
goto :eof

REM Manual retraining
:retrain_model
if "%~2"=="" (
    call :print_error "Model name is required for retraining"
    echo Usage: %0 retrain ^<model_name^> [reason]
    exit /b 1
)
set "model_name=%~2"
set "reason=%~3"
if "%reason%"=="" set "reason=Manual retraining"

call :print_info "Retraining model: !model_name! (Reason: !reason!)"
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --retrain "!model_name!"
goto :eof

REM Retrain all models
:retrain_all
call :print_info "Retraining all models..."
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --retrain-all
goto :eof

REM Check performance degradation
:check_performance
call :print_info "Checking for performance degradation..."
call :check_python
call :check_scheduler_script

%PYTHON_CMD% "%SCHEDULER_SCRIPT%" --check-performance
goto :eof

REM Show help
:show_help
echo Model Retraining Scheduler Management Script (Windows)
echo.
echo Usage: %0 ^<command^> [options]
echo.
echo Commands:
echo   start                 Start the retraining scheduler
echo   stop                  Stop the retraining scheduler
echo   status                Show scheduler status
echo   retrain ^<model^>       Manually retrain a specific model
echo   retrain-all           Retrain all available models
echo   check-performance     Check for performance degradation
echo   help                  Show this help message
echo.
echo Examples:
echo   %0 start
echo   %0 retrain random_forest
echo   %0 retrain-all
echo   %0 check-performance
echo   %0 status
echo.
echo Available models: random_forest, xgboost, lightgbm, ensemble, ols_static
goto :eof

REM Main script logic
if "%1"=="" goto show_help
if "%1"=="start" goto start_scheduler
if "%1"=="stop" goto stop_scheduler
if "%1"=="status" goto status_scheduler
if "%1"=="retrain" goto retrain_model
if "%1"=="retrain-all" goto retrain_all
if "%1"=="check-performance" goto check_performance
if "%1"=="help" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help

call :print_error "Unknown command: %1"
echo.
goto show_help