#!/bin/bash
# Полный рендер обеих версий видео этапа 2 на машине с видеокартой (Linux / macOS).
# Запуск из корня репозитория:  bash stage2_logistics/run_render_gpu.sh [samples=16] [step=all|test|render|compose] [edition=all|ru|en]
set -e
cd "$(dirname "$0")/.."
SAMPLES=${1:-16}; STEP=${2:-all}; EDITION=${3:-all}
case "$STEP" in all|test|render|compose) ;; *) echo "step: all | test | render | compose"; exit 1;; esac
case "$EDITION" in all) EDITIONS="ru en";; ru|en) EDITIONS="$EDITION";; *) echo "edition: all | ru | en"; exit 1;; esac
if [ ! -d .venv ]; then
  echo "[1/5] Создаю окружение Python 3.11 и ставлю bpy 4.5, numpy, pillow (~400 МБ)..."
  python3.11 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install "bpy==4.5.*" numpy pillow
fi
PY=.venv/bin/python
command -v ffmpeg >/dev/null || { echo "ffmpeg не найден в PATH"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe не найден в PATH (нужен монтажу): установите полный ffmpeg"; exit 1; }
if [ "$STEP" = all ] || [ "$STEP" = test ]; then
  for ED in $EDITIONS; do
    echo "[2/5] Пробный кадр версии $ED (в логе должно быть имя видеокарты, не llvmpipe)..."
    $PY stage2_logistics/build2.py --edition "$ED" --shot S2_15_scan --res 1920 1080 --samples "$SAMPLES" 2>&1 | tee "/tmp/stage2_test_$ED.log" | grep -E "\[build2\] edition|renderer|\[render\] frame|Error"
    if grep -qiE "llvmpipe|software" "/tmp/stage2_test_$ED.log"; then echo "EEVEE считает на процессоре: нужен драйвер с EGL (NVIDIA >= 470); НЕ используйте xvfb-run. Запасной путь: --engine CYCLES --gpu OPTIX (RENDER.md, п.8)"; exit 1; fi
    echo "Кадр: stage2_logistics/out/stills_$ED/ — проверьте картинку: надписи на языке версии, логотип."
  done
fi
if [ "$STEP" = all ] || [ "$STEP" = render ]; then
  for ED in $EDITIONS; do
    echo "[3/5] Рендер кадров версии $ED (возобновляемый)..."
    $PY stage2_logistics/build2.py --edition "$ED" --render --samples "$SAMPLES"
    echo "[4/5] Проверка комплектности версии $ED..."
    $PY stage2_logistics/edl.py --edition "$ED" --check-frames "stage2_logistics/out/frames_$ED"
  done
fi
if [ "$STEP" = all ] || [ "$STEP" = compose ]; then
  for ED in $EDITIONS; do
    echo "[5/5] Монтаж версии $ED..."
    $PY stage2_logistics/post/compose2.py --edition "$ED" --final
  done
  echo "Готово: stage2_logistics/deliverables/demo_full_cycle_<версия>_<логотип>_1080p.mp4 (+ _compact.mp4)"
fi
