@echo off
title TiltRadar — My Progress
cd /d C:\Users\boure\tilt-radar
call C:\Users\boure\anaconda3\Scripts\activate.bat C:\Users\boure\anaconda3\envs\tilt-radar
python scripts/my_progress.py --summoner "MarreDesNoobsNul" --tag "007"
pause
