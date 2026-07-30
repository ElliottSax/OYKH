@echo off
REM Force delete training images - Run as Administrator if needed

echo Unlocking and enabling delete...
echo.

cd /d "C:\projects\oykh-temp\lora-training-v2\images"

REM Remove read-only attribute from all PNG files
attrib -r *.png

echo Files unlocked!
echo.
echo You can now delete images in Windows Explorer
echo.
echo OR delete specific images with commands like:
echo   del train_001_pointing_forward.png
echo   del train_015_*.png
echo.
pause
