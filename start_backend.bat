@echo off
title TiltRadar — Backend
cd /d C:\Users\boure\tilt-radar
call C:\Users\boure\anaconda3\Scripts\activate.bat C:\Users\boure\anaconda3\envs\tilt-radar
uvicorn api.main:app --host 0.0.0.0 --port 8002
pause
