# Полный рендер этапа 2 на машине с видеокартой (Windows, PowerShell).
# Запуск из корня репозитория:  powershell -ExecutionPolicy Bypass -File stage2_logistics\run_render_gpu.ps1 [-Samples 16] [-Step test|render|compose]
# Файл хранится в UTF-8 с BOM: без BOM Windows PowerShell 5.1 читает его как ANSI и не может разобрать русские строки.
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
$env:PYTHONIOENCODING = "utf-8"

# Ноутбук с двумя видеокартами (NVIDIA Optimus): python.exe по умолчанию получает встроенную графику Intel, и кадр
# считается ~190 с вместо ~7 с. Переменная действует только на процессы, запущенные из этого скрипта; на машинах
# без Optimus она ни на что не влияет.
if (-not $env:SHIM_MCCOMPAT) { $env:SHIM_MCCOMPAT = "0x800000001" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg не найден в PATH: установите (winget install Gyan.FFmpeg) и откройте новое окно PowerShell" }
# Монтажу нужен ещё и ffprobe, а в PATH бывает только ffmpeg.exe. Найденную папку добавляем в КОНЕЦ PATH этого
# процесса, чтобы ffmpeg остался прежним. Проверяем до рендера, а не после нескольких часов счёта.
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    $dirs = @("$env:LOCALAPPDATA\Microsoft\WinGet\Links", "$env:ProgramFiles\ffmpeg\bin", "C:\ffmpeg\bin",
              "$env:ProgramData\chocolatey\bin", "$env:USERPROFILE\scoop\shims", "$env:ProgramFiles\DownloadHelper CoApp")
    $probe = $dirs | Where-Object { Test-Path (Join-Path $_ "ffprobe.exe") } | Select-Object -First 1
    if (-not $probe) { throw "ffprobe не найден ни в PATH, ни в типичных папках: установите полный ffmpeg (winget install Gyan.FFmpeg) и откройте новое окно PowerShell" }
    $env:PATH = "$env:PATH;$probe"
    Write-Host "ffprobe взят из $probe (добавлен в PATH только для этого запуска)"
}

if ($Step -in @("all", "test")) {
    Write-Host "[2/5] Пробный кадр (в логе должно быть имя вашей дискретной видеокарты, не llvmpipe и не Intel)..."
    & $py stage2_logistics\build2.py --shot S2_08_load --res 1920 1080 --samples $Samples | Tee-Object -Variable log | Select-String -Pattern "renderer|\[render\] frame|Error"
    if ($LASTEXITCODE -ne 0) { throw "Пробный кадр не отрендерился (код $LASTEXITCODE), см. сообщения выше" }
    $renderer = "$($log | Select-String -Pattern 'renderer:' | Select-Object -First 1)"
    if ($renderer -match "llvmpipe|Software") { throw "EEVEE считает на процессоре (llvmpipe). Обновите драйвер видеокарты или используйте --engine CYCLES --gpu OPTIX (см. RENDER.md, п.8)" }
    $discrete = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA|GeForce|Quadro|Radeon RX" } | Select-Object -First 1
    if ($discrete -and $renderer -match "Intel" -and $renderer -notmatch "Arc") { throw "EEVEE считает на встроенной графике, хотя в системе есть $($discrete.Name). Назначьте python.exe высокую производительность: Параметры Windows > Дисплей > Графика (см. RENDER.md, п.8)" }
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
    if ($LASTEXITCODE -ne 0) { throw "Монтаж завершился с ошибкой (код $LASTEXITCODE), см. сообщения выше" }
    Write-Host "Готово: stage2_logistics\deliverables\demo_full_cycle_1080p.mp4 (+ _compact.mp4)"
}
