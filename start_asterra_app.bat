@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ASTERRA one-click launcher: backend + frontend + readiness checks.
set "ROOT_DIR=%~dp0"
set "FRONTEND_DIR=%ROOT_DIR%frontend"
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "BACKEND_URL=http://127.0.0.1:8000/api/health"
set "FRONTEND_URL=http://127.0.0.1:5173"

rem SegFormer is the primary semantic backend. Download the checkpoint once
rem into the configured cache and reuse it; failed loads remain a labelled
rem heuristic fallback so terrain reconstruction still completes.
if not defined ASTERRA_SEGMENTATION_LOCAL_ONLY set "ASTERRA_SEGMENTATION_LOCAL_ONLY=0"
if not defined ASTERRA_PRELOAD_MODEL set "ASTERRA_PRELOAD_MODEL=1"
if not defined ASTERRA_SEGMENTATION_TIMEOUT set "ASTERRA_SEGMENTATION_TIMEOUT=60"
if not defined ASTERRA_SEGMENTATION_CACHE set "ASTERRA_SEGMENTATION_CACHE=%ROOT_DIR%backend\runtime\segmentation_cache"

if not exist "%ROOT_DIR%backend\main.py" goto :missing_backend
if not exist "%FRONTEND_DIR%\package.json" goto :missing_frontend

if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
  call :check_python "%ROOT_DIR%.venv\Scripts\python.exe"
  if not errorlevel 1 set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
  if errorlevel 1 echo [WARN] The project virtual environment exists but its Python/dependencies are unavailable. Continuing with system Python discovery.
)
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE (
  call :check_python "%%P"
  if not errorlevel 1 set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
  where py >nul 2>&1
  if not errorlevel 1 py -3 -c "import fastapi,uvicorn,multipart,pydantic,numpy,scipy,rasterio,torch,PIL,trimesh,transformers" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
  )
)
rem A moved/removed base interpreter can leave .venv\Scripts\python.exe
rem present but unusable. Reuse its installed packages with a compatible
rem system Python before reporting that the backend is unavailable.
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
  call :check_python_with_project_packages "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  if not errorlevel 1 (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    set "PYTHON_ARGS="
    set "PYTHONPATH=%ROOT_DIR%.venv\Lib\site-packages"
  )
)
if not defined PYTHON_EXE if exist "%ProgramFiles%\Python311\python.exe" (
  call :check_python_with_project_packages "%ProgramFiles%\Python311\python.exe"
  if not errorlevel 1 (
    set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
    set "PYTHON_ARGS="
    set "PYTHONPATH=%ROOT_DIR%.venv\Lib\site-packages"
  )
)
if not defined PYTHON_EXE goto :missing_python

if not exist "%FRONTEND_DIR%\node_modules\.bin\vite.cmd" (
  echo [INFO] Installing frontend dependencies...
  pushd "%FRONTEND_DIR%"
  call npm.cmd ci
  if errorlevel 1 (
    popd
    goto :startup_failed
  )
  popd
)

call :probe_url "%BACKEND_URL%"
if errorlevel 1 (
  echo [INFO] Starting FastAPI backend...
  if defined PYTHON_ARGS (
    start "ASTERRA Backend" /D "%ROOT_DIR%" "%ComSpec%" /k "%PYTHON_EXE% %PYTHON_ARGS% -m backend.main"
  ) else (
    start "ASTERRA Backend" /D "%ROOT_DIR%" "%ComSpec%" /k ""%PYTHON_EXE%" -m backend.main"
  )
) else echo [INFO] Backend already running.

call :wait_for_url "%BACKEND_URL%" "FastAPI backend" 120
if errorlevel 1 goto :startup_failed

call :probe_url "%FRONTEND_URL%"
if errorlevel 1 (
  echo [INFO] Starting Vite frontend...
  start "ASTERRA Frontend" /D "%FRONTEND_DIR%" "%ComSpec%" /k npm.cmd run dev -- --host 127.0.0.1
) else echo [INFO] Frontend already running.

call :wait_for_url "%FRONTEND_URL%" "Vite frontend" 60
if errorlevel 1 goto :startup_failed

echo.
echo [READY] ASTERRA backend and frontend are running.
start "" "%FRONTEND_URL%"
goto :startup_success

:check_python
"%~1" -c "import fastapi,uvicorn,multipart,pydantic,numpy,scipy,rasterio,torch,PIL,trimesh,transformers" >nul 2>&1
exit /b %errorlevel%

:check_python_with_project_packages
set "CHECK_PYTHONPATH=%ROOT_DIR%.venv\Lib\site-packages"
set "PYTHONPATH=%CHECK_PYTHONPATH%;%PYTHONPATH%"
"%~1" -c "import fastapi,uvicorn,multipart,pydantic,numpy,scipy,rasterio,torch,PIL,trimesh,transformers" >nul 2>&1
exit /b %errorlevel%

:probe_url
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { $r=Invoke-WebRequest -UseBasicParsing -Uri '%~1' -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } } catch {}; exit 1" >nul 2>&1
exit /b %errorlevel%

:wait_for_url
set "WAIT_URL=%~1"
set "WAIT_NAME=%~2"
set "WAIT_SECONDS=%~3"
echo [INFO] Waiting for %WAIT_NAME%...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; $url='%WAIT_URL%'; $limit=[int]'%WAIT_SECONDS%'; for($i=0;$i -lt $limit;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] %WAIT_NAME% did not become ready within %WAIT_SECONDS% seconds.
  exit /b 1
)
echo [INFO] %WAIT_NAME% is ready.
exit /b 0

:missing_backend
echo [ERROR] Backend entry point not found: %ROOT_DIR%backend\main.py
goto :startup_failed

:missing_frontend
echo [ERROR] Frontend package.json not found: %FRONTEND_DIR%\package.json
goto :startup_failed

:missing_python
echo [ERROR] A compatible Python with the complete ASTERRA backend dependency set was not found.
echo [INFO] Python discovery checked the project virtual environment, system Python, and the Python launcher.
if exist "%ROOT_DIR%.venv\pyvenv.cfg" echo [INFO] If .venv points to a removed Python installation, recreate it with the commands below.
echo Install once with:
echo   py -3.11 -m venv "%ROOT_DIR%.venv"
echo   "%ROOT_DIR%.venv\Scripts\python.exe" -m pip install -r "%ROOT_DIR%backend\requirements.txt"
echo   python -m pip install -r "%ROOT_DIR%backend\requirements.txt"
goto :startup_failed

:startup_failed
echo.
echo [ERROR] ASTERRA did not start completely.
pause
endlocal
exit /b 1

:startup_success
endlocal
exit /b 0
