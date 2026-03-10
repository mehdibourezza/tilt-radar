@echo off
title TiltRadar — Agent (MarreDesNoobsNul#007)
cd /d C:\Users\boure\tilt-radar
call C:\Users\boure\anaconda3\Scripts\activate.bat C:\Users\boure\anaconda3\envs\tilt-radar
python -m agent.local_agent --summoner "MarreDesNoobsNul" --tag "007" --server ws://localhost:8002
pause
