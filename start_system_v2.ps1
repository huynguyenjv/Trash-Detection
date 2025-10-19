# Start Waste Detection System - Backend V2

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Starting Waste Detection System - Backend V2" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green

# Start backend
Write-Host "`nStarting backend server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd waste-system\backend-v2; python backend.py"

Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start frontend
Write-Host "`nStarting frontend dev server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd waste-system\frontend; npm run dev"

Write-Host "Waiting for frontend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "System is ready!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green

Write-Host "`nURLs:" -ForegroundColor Yellow
Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor White

Write-Host "`nFeatures:" -ForegroundColor Yellow
Write-Host "  - Multi-object detection (detect NHIEU objects/frame)" -ForegroundColor White
Write-Host "  - Realtime WebSocket detection" -ForegroundColor White
Write-Host "  - YOLOv8n default model (80 COCO classes)" -ForegroundColor White
Write-Host "  - Waste classification (organic/recyclable/hazardous/other)" -ForegroundColor White

Write-Host "`nOpening browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process "http://localhost:5173"

Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

