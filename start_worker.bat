@echo off
title TiltRadar — Celery Worker
cd /d C:\Users\boure\tilt-radar
call C:\Users\boure\anaconda3\Scripts\activate.bat C:\Users\boure\anaconda3\envs\tilt-radar
celery -A workers.celery_app worker --loglevel=info --pool=solo
pause
