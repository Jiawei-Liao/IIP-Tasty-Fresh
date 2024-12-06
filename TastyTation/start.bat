@echo off

:: Get the absolute path of the script's directory
set "SCRIPT_DIR=%~dp0"

:: Start the backend server in a new terminal with virtual environment activated
start cmd.exe /K "cd /d "%SCRIPT_DIR%backend" && conda activate iip && python ./endpoints.py"

:: Start the frontend server in a new terminal
start cmd.exe /K "cd /d "%SCRIPT_DIR%frontend" && npm start"