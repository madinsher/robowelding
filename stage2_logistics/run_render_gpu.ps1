# Полный рендер обеих версий видео этапа 2 на машине с видеокартой (Windows, PowerShell).
# Запуск из корня репозитория:
#   powershell -ExecutionPolicy Bypass -File stage2_logistics\run_render_gpu.ps1 [-Edition ru|en|all] [-Samples 16] [-Step test|render|compose]
# Файл хранится в UTF-8 с BOM: без BOM Windows PowerShell 5.1 читает его как ANSI и не может разобрать русские строки.
param([int]$Samples = 16, [string]$Step = "all", [string]$Edition = "all")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))   # корень репозитория
if ($Step -notin @("all", "test", "render", "compose")) { throw "-Step: all | test | render | compose" }
if ($Edition -notin @("all", "ru", "en")) { throw "-Edition: all | ru | en" }
$editions = if ($Edition -eq "all") { @("ru", "en") } else { @($Edition) }

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
    foreach ($ed in $editions) {
        Write-Host "[2/5] Пробный кадр версии $ed (в логе должно быть имя вашей дискретной видеокарты, не llvmpipe и не Intel)..."
        & $py stage2_logistics\build2.py --edition $ed --shot S2_15_scan --res 1920 1080 --samples $Samples | Tee-Object -Variable log | Select-String -Pattern "\[build2\] edition|renderer|\[render\] frame|Error"
        if ($LASTEXITCODE -ne 0) { throw "Пробный кадр версии $ed не отрендерился (код $LASTEXITCODE), см. сообщения выше" }
        $renderer = "$($log | Select-String -Pattern 'renderer:' | Select-Object -First 1)"
        if ($renderer -match "llvmpipe|Software") { throw "EEVEE считает на процессоре (llvmpipe). Обновите драйвер видеокарты или используйте --engine CYCLES --gpu OPTIX (см. RENDER.md, п.8)" }
        $discrete = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA|GeForce|Quadro|Radeon RX" } | Select-Object -First 1
        if ($discrete -and $renderer -match "Intel" -and $renderer -notmatch "Arc") { throw "EEVEE считает на встроенной графике, хотя в системе есть $($discrete.Name). Назначьте python.exe высокую производительность: Параметры Windows > Дисплей > Графика (см. RENDER.md, п.8)" }
        Write-Host "Кадр: stage2_logistics\out\stills_$ed\frame_*.png — откройте и проверьте картинку: надписи на языке версии, логотип."
    }
}
if ($Step -in @("all", "render")) {
    foreach ($ed in $editions) {
        Write-Host "[3/5] Рендер кадров версии $ed (возобновляемый: повторный запуск пропускает готовые)..."
        & $py stage2_logistics\build2.py --edition $ed --render --samples $Samples
        Write-Host "[4/5] Проверка комплектности кадров версии $ed..."
        & $py stage2_logistics\edl.py --edition $ed --check-frames stage2_logistics\out\frames_$ed
        if ($LASTEXITCODE -ne 0) { throw "Не все кадры версии $ed на месте — запустите скрипт ещё раз с -Step render -Edition $ed" }
    }
}
if ($Step -in @("all", "compose")) {
    foreach ($ed in $editions) {
        Write-Host "[5/5] Монтаж версии $ed с титрами, логотипом и звуком..."
        & $py stage2_logistics\post\compose2.py --edition $ed --final
        if ($LASTEXITCODE -ne 0) { throw "Монтаж версии $ed завершился с ошибкой (код $LASTEXITCODE), см. сообщения выше" }
        Write-Host "[5/5] Промежуточная склейка версии $ed (без титров, логотипов и звука)..."
        & $py stage2_logistics\post\compose2.py --edition $ed --raw
        if ($LASTEXITCODE -ne 0) { throw "Промежуточная склейка версии $ed завершилась с ошибкой (код $LASTEXITCODE), см. сообщения выше" }
    }
    Write-Host "Готово: stage2_logistics\deliverables\demo_full_cycle_<версия>_<логотип>_1080p.mp4 (+ _compact.mp4, _raw.mp4)"
}
