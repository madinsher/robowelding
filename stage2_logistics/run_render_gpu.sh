#!/bin/bash
# Полный рендер этапа 2 на машине с видеокартой (Linux / macOS).
# Запуск из корня репозитория:  bash stage2_logistics/run_render_gpu.sh [samples=16] [step=all|test|render|compose]
set -e
cd "$(dirname "$0")/.."
SAMPLES=${1:-16}; STEP=${2:-all}
if [ ! -d .venv ]; then
  echo "[1/5] Создаю окружение Python 3.11 и ставлю bpy 4.5, numpy, pillow (~400 МБ)..."
  python3.11 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install "bpy==4.5.*" numpy pillow
fi
PY=.venv/bin/python
command -v ffmpeg >/dev/null || { echo "ffmpeg не найден в PATH"; exit 1; }
if [ "$STEP" = all ] || [ "$STEP" = test ]; then
  echo "[2/5] Пробный кадр (в логе должно быть имя видеокарты, не llvmpipe)..."
  $PY stage2_logistics/build2.py --shot S2_08_load --res 1920 1080 --samples "$SAMPLES" 2>&1 | tee /tmp/stage2_test.log | grep -E "renderer|\[render\] frame|Error"
  if grep -qiE "llvmpipe|software" /tmp/stage2_test.log; then echo "EEVEE считает на процессоре: нужен драйвер с EGL (NVIDIA >= 470); НЕ используйте xvfb-run. Запасной путь: --engine CYCLES --gpu OPTIX (RENDER.md, п.8)"; exit 1; fi
  echo "Кадр: stage2_logistics/out/stills/ — проверьте картинку."
fi
if [ "$STEP" = all ] || [ "$STEP" = render ]; then
  echo "[3/5] Рендер новых кадров (возобновляемый)..."
  $PY stage2_logistics/build2.py --render --samples "$SAMPLES"
  echo "[4/5] Проверка комплектности..."
  $PY stage2_logistics/edl.py --check-frames stage2_logistics/out/frames
fi
if [ "$STEP" = all ] || [ "$STEP" = compose ]; then
  echo "[5/5] Монтаж итогового видео..."
  $PY stage2_logistics/post/compose2.py --frames stage2_logistics/out/frames --out stage2_logistics/deliverables/demo_full_cycle_1080p.mp4 --compact stage2_logistics/deliverables/demo_full_cycle_1080p_compact.mp4 --verify
  echo "Готово: stage2_logistics/deliverables/demo_full_cycle_1080p.mp4 (+ _compact.mp4)"
fi
