@echo off
setlocal
cd /d "%~dp0"
if not exist "data\character_print.pt" goto missing
if not exist "data\og_emb.npy" goto missing
if not exist "data\character_head.npz" goto missing
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 goto error
:install
if exist ".venv\dependencies-installed" goto serve
".venv\Scripts\python.exe" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
type nul > ".venv\dependencies-installed"
:serve
echo Open http://127.0.0.1:8000 when the server is ready.
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto error
exit /b 0
:missing
echo Trained model files are missing from data.
goto error
:error
echo See README.md and the error above.
pause
exit /b 1
