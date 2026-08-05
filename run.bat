@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Face Age Gender Emotion Detection
echo.
echo === Real-Time Face / Age / Gender / Emotion ===
echo Working directory: %CD%
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PY=py -3.11"
  %PY% -c "import sys" >nul 2>&1
  if errorlevel 1 set "PY=py -3"
) else (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PY=python"
  ) else (
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.11 64-bit from https://www.python.org/downloads/
    echo Check "Add python.exe to PATH", then delete the venv folder and re-run.
    pause
    exit /b 1
  )
)

if not exist "main.py" (
  echo ERROR: main.py not found.
  echo Put this whole repo folder on your Desktop and run run.bat from inside it.
  pause
  exit /b 1
)

if not exist "mivolo\__init__.py" (
  echo ERROR: Vendored mivolo package missing.
  echo Re-clone the full repository.
  pause
  exit /b 1
)

%PY% -c "import sys; v=sys.version_info; print('Using Python %d.%d.%d'%v[:3]); raise SystemExit(0 if (v.major==3 and v.minor in (10,11)) else 1)"
if errorlevel 1 (
  echo.
  echo ERROR: Use Python 3.10 or 3.11 64-bit on Windows.
  echo Python 3.12/3.13 often break TensorFlow / numpy wheels.
  echo 1^) Install Python 3.11 from python.org
  echo 2^) Delete the "venv" folder in this repo
  echo 3^) Run run.bat again
  pause
  exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
  echo Creating virtual environment...
  %PY% -m venv venv
  if errorlevel 1 (
    echo ERROR: Failed to create venv.
    pause
    exit /b 1
  )
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Failed to activate venv.
  pause
  exit /b 1
)

python -c "import cv2, torch, transformers, numpy, mivolo" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Installing dependencies from wheels ^(no Visual Studio needed^)...
  echo This can take several minutes on first run.
  echo.

  python -m pip install --upgrade pip wheel
  if errorlevel 1 goto :install_fail

  python -m pip install "setuptools>=68,<81"
  if errorlevel 1 goto :install_fail

  REM Prefer binary wheels; never compile numpy/scipy from source
  python -m pip install --only-binary=:all: numpy==1.26.4 scipy==1.11.4 pillow==10.4.0
  if errorlevel 1 (
    echo Failed binary install of numpy/scipy. Wrong Python version?
    goto :install_fail
  )

  python -m pip install --only-binary=:all: opencv-python==4.10.0.84
  python -m pip install --only-binary=:all: opencv-contrib-python==4.10.0.84
  if errorlevel 1 goto :install_fail

  python -m pip install --only-binary=:all: torch==2.2.2 torchvision==0.17.2
  if errorlevel 1 goto :install_fail

  python -m pip install --only-binary=:all: tensorflow==2.15.1 onnxruntime==1.17.3
  if errorlevel 1 goto :install_fail

  python -m pip install transformers==4.51.0 accelerate==0.33.0 "timm==0.8.13.dev0" huggingface_hub
  if errorlevel 1 goto :install_fail

  python -c "import cv2, torch, transformers, numpy, mivolo, tensorflow" >nul 2>&1
  if errorlevel 1 (
    echo.
    echo Verifying imports failed. Trying requirements.txt as fallback...
    python -m pip install --prefer-binary -r requirements.txt
    if errorlevel 1 goto :install_fail
  )

  echo.
  echo Dependencies installed.
)

if exist "scripts\download_models.py" (
  echo.
  echo Checking local model files...
  python scripts\download_models.py
)

echo.
echo Starting app...
echo Press Q in the video window to quit.
echo.
python main.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo App exited with code %EXITCODE%.
  pause
)
endlocal
exit /b %EXITCODE%

:install_fail
echo.
echo ERROR: Dependency install failed.
echo Fix:
echo   1. Use Python 3.11 64-bit from python.org
echo   2. Delete the venv folder
echo   3. Run run.bat again
echo Do NOT install Visual Studio just for this app — packages should use wheels.
pause
endlocal
exit /b 1
