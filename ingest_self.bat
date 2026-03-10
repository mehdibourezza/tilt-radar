@echo off
title TiltRadar — Ingest Self History
cd /d C:\Users\boure\tilt-radar
call C:\Users\boure\anaconda3\Scripts\activate.bat C:\Users\boure\anaconda3\envs\tilt-radar
python scripts/ingest_self.py
pause
