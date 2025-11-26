@echo off
echo Starting Zomato ETA Prediction System...

start "Zomato Backend" cmd /k "cd backend && pip install -r requirements.txt && python main.py"
timeout /t 10
start "Zomato Frontend" cmd /k "cd frontend && set PORT=3001 && npm start"

echo System started! 
echo Backend running on http://localhost:8001
echo Frontend running on http://localhost:3001
