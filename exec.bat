@echo off
chcp 65001 >nul 2>&1
title Intra-Class Variation EDA Pipeline

echo ============================================================
echo   INTRA-CLASS VARIATION EDA PIPELINE
echo ============================================================
echo.
echo   Which pipeline do you want to run?
echo.
echo   1. Pre-Annotation (DINOv2)               (frame quality assessment - BEFORE annotation)
echo   2. Post-Annotation and Pre-Training      (intra-class analysis - AFTER annotation and BEFORE training)
echo.
echo   0. Exit
echo.
echo ============================================================

set /p choice="  Select [0-2]: "

if "%choice%"=="1" (
    echo.
    echo   Launching Pre-Annotation pipeline (DINOv2^)...
    echo.
    python "master_scripts/master_script_dinov2_PreAnn.py"
) else if "%choice%"=="2" (
    echo.
    echo   Launching Post-Annotation and Pre-Training pipeline...
    echo.
    python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py"
) else if "%choice%"=="0" (
    echo.
    echo   Bye!
    exit /b 0
) else (
    echo.
    echo   Invalid choice. Please run again.
)

echo.
pause
