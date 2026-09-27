# Полный рендер этапа 2 на машине с видеокартой (Windows, PowerShell).
# Запуск из корня репозитория:  powershell -ExecutionPolicy Bypass -File stage2_logistics\run_render_gpu.ps1 [-Samples 16] [-Step test|render|compose]
param([int]$Samples = 16, [string]$Step = "all")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))   # корень репозитория

if (-not (Test-Path .venv)) {
    Write-Host "[1/5] Создаю окружение Python 3.11 и ставлю bpy 4.5, numpy, pillow (~400 МБ)..."
    py -3.11 -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install "bpy==4.5.*" numpy pillow
}
$py = ".venv\Scripts\python.exe"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg не найден в PATH: установите (winget install Gyan.FFmpeg) и откройте новое окно PowerShell" }

if ($Step -in @("all", "test")) {
    Write-Host "[2/5] Пробный кадр (в логе должно быть имя вашей видеокарты, не llvmpipe)..."
    & $py stage2_logistics\build2.py --shot S2_08_load --res 1920 1080 --samples $Samples 2>&1 | Tee-Object -Variable log | Select-String -Pattern "renderer|\[render\] frame|Error"
    if ($log -match "llvmpipe|Software") { throw "EEVEE считает на процессоре (llvmpipe). Обновите драйвер видеокарты или используйте --engine CYCLES --gpu OPTIX (см. RENDER.md, п.8)" }
    Write-Host "Кадр: stage2_logistics\out\stills\frame_*.png — откройте и проверьте картинку."
}
if ($Step -in @("all", "render")) {
    Write-Host "[3/5] Рендер новых кадров монтажа (возобновляемый: повторный запуск пропускает готовые)..."
    & $py stage2_logistics\build2.py --render --samples $Samples
    Write-Host "[4/5] Проверка комплектности кадров..."
    & $py stage2_logistics\edl.py --check-frames stage2_logistics\out\frames
    if ($LASTEXITCODE -ne 0) { throw "Не все кадры на месте — запустите скрипт ещё раз с -Step render" }
}
if ($Step -in @("all", "compose")) {
    Write-Host "[5/5] Монтаж итогового видео с титрами и звуком..."
    & $py stage2_logistics\post\compose2.py --frames stage2_logistics\out\frames --out stage2_logistics\deliverables\demo_full_cycle_1080p.mp4 --compact stage2_logistics\deliverables\demo_full_cycle_1080p_compact.mp4 --verify
    Write-Host "Готово: stage2_logistics\deliverables\demo_full_cycle_1080p.mp4 (+ _compact.mp4)"
}
