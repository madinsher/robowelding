# Инструкция: итоговый рендер и монтаж видео этапа 2 на машине с видеокартой

Итог — `stage2_logistics/deliverables/demo_full_cycle_1080p.mp4` (1920×1080, 24 fps, ≈ 90 с, русские титры,
звук) и компактная версия. **Сварку заново не рендерим**: монтаж берёт готовые отрезки видео этапа 1
(`demo_video/deliverables/demo_welding_cell_1080p.mp4`, кадры 97–456 и 541–930) и вставляет их между новыми
сценами. Рендерятся только новые планы этапа 2 — список кадров задаёт монтажный лист `stage2_logistics/edl.py`
(сейчас 1402 кадра; точное число печатает `python edl.py`).

Схема работы:

```
build2.py --render        новые кадры  stage2_logistics/out/frames/frame_NNNN.png   (NNNN = кадр сцены этапа 2)
        +
demo_welding_cell_1080p.mp4   готовые кадры сварки этапа 1 (не рендерятся)
        |
post/compose2.py          монтаж по post/edl.json + титры (post/storyboard2.json) + звук -> mp4
```

## 1. Что нужно на машине

* Видеокарта с поддержкой OpenGL 4.3 (NVIDIA / AMD / Intel Arc), свежий драйвер, 6+ ГБ видеопамяти.
* Git, ffmpeg в `PATH` (`ffmpeg -version` должен работать).
* **Python 3.11** (именно 3.11 — под него собран модуль `bpy` 4.5) **или** Blender 4.5 LTS (вариант Б ниже).
* ~15 ГБ свободного места (PNG 1080p ≈ 4–6 МБ на кадр).

## 2. Установка

Вариант А (рекомендуется, так же считался этап 1) — Python 3.11 + модуль `bpy`:

```bash
git clone <репозиторий> robowelding && cd robowelding
git checkout claude/stage2-logistics-render-3zd849
# Windows (PowerShell):
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install "bpy==4.5.*" numpy pillow
.venv\Scripts\Activate.ps1          # если PowerShell запрещает скрипты: Set-ExecutionPolicy -Scope Process Bypass
# Linux / macOS:
python3.11 -m venv .venv && . .venv/bin/activate
pip install "bpy==4.5.*" numpy pillow
```

Вариант Б — приложение Blender 4.5 LTS. В его встроенный Python нужно поставить Pillow (им рисуется маркировка
на деталях):

```bash
"<папка Blender>/4.5/python/bin/python3.11" -m pip install pillow        # Linux/macOS
"<папка Blender>\4.5\python\bin\python.exe" -m pip install pillow         # Windows
```

Дальше в командах **рендера** вместо `python stage2_logistics/build2.py ARGS` пишите
`blender -b --python stage2_logistics/build2.py -- ARGS` (аргументы скрипта — после `--`). Остальные скрипты
(`edl.py`, `post/compose2.py`, `post/storyboard2.py`, проверки `tests/t_*2.py` без Blender-рендера) запускайте
встроенным Python Blender'а — в нём есть numpy, а Pillow вы поставили выше:
`"<папка Blender>/4.5/python/bin/python3.11" stage2_logistics/post/compose2.py ...` (Windows:
`"<папка Blender>\4.5\python\bin\python.exe" ...`).

Шрифты для подписей, табличек и маркировки лежат в репозитории (`stage2_logistics/assets/fonts`), ставить их не нужно.

## 3. Пробный кадр и проверка, что считает видеокарта

```bash
python stage2_logistics/build2.py --shot S2_08_load --res 1920 1080 --samples 16
```

(`--shot ИМЯ` без `--render` рендерит середину плана; имена планов — `python stage2_logistics/edl.py`, любой кадр
сцены — `--still N`).

* Первый запуск дольше (≈ 3–5 мин на CPU): строится траектория сварки этапа 1 (кэш `demo_video/out/anim_cache_*.npz`).
  Хореография этапа 2 уже посчитана и лежит в репозитории (`stage2_logistics/data/plan2_*.npz`, берётся, если код
  хореографии не менялся). Следующие запуски берут кэш, сборка сцены ≈ 2–3 мин.
* Кадр появится в `stage2_logistics/out/stills/frame_NNNN.png` (погрузчик ставит спул на планшайбу).
* В логе есть строка `[render] engine BLENDER_EEVEE_NEXT, OpenGL/Vulkan renderer: ...` — там должно быть имя вашей
  видеокарты (например `NVIDIA GeForce RTX ...`). Если там `llvmpipe` / `Software` — рендер идёт на процессоре,
  см. раздел 8.
* Строка `[render] frame NNNN ... NN.Ns` — время кадра. Первый кадр включает компиляцию шейдеров (до 1–2 мин),
  дальше на современной видеокарте ожидается ≈ 2–6 с на кадр 1080p.

## 4. (Необязательно) быстрый черновик всего ролика

```bash
python stage2_logistics/build2.py --preview                 # каждый 6-й кадр монтажа, 640×360
python stage2_logistics/post/compose2.py --frames stage2_logistics/out/preview_frames --out stage2_logistics/out/preview.mp4 --preview
```

Черновик проверяет монтаж, титры и звук до долгого рендера (пропущенные кадры «держатся» предыдущими).

## 5. Финальный рендер новых кадров

```bash
python stage2_logistics/build2.py --render --samples 16
```

* Кадры пишутся в `stage2_logistics/out/frames/`. Рендер **возобновляемый**: при повторном запуске готовые кадры
  пропускаются — можно прервать (Ctrl+C) и продолжить той же командой.
* Настройки рендера — как в этапе 1 (EEVEE, AgX, тени от ключевого света и дуги), чтобы новые планы по картинке
  совпадали с вставленным видео сварки. `--samples 16` убирает шум сглаживания (в этапе 1 было 8 из-за CPU);
  можно 24–32, если время позволяет.
* `--quality high` включает тени от всех источников и screen-space отражения — картинка богаче, но заметнее
  отличается от вставок этапа 1. По умолчанию не использовать.
* Несколько машин: разделите работу — `--worker 0/2` на одной и `--worker 1/2` на другой, затем сложите PNG в одну
  папку. На одной машине запускайте один процесс: EEVEE и так загружает видеокарту целиком (выбора GPU для
  EEVEE по процессам нет; для Cycles можно задать `CUDA_VISIBLE_DEVICES`).
* Папка кадров помечена хэшем хореографии (`out/frames/.plan_id`). Если после правок хореографии/монтажа
  запустить `--render` в ту же папку, `build2.py` откажется смешивать старые и новые кадры — перенесите старые
  кадры или укажите другую `--outdir`.
* Только один план (например, после правки камеры): `--shot S2_08_load --render` (удалите старые кадры этого
  плана, иначе они будут пропущены). Имена планов — `stage2_logistics/cameras2.py`.

Проверка комплектности:

```bash
python stage2_logistics/edl.py --check-frames stage2_logistics/out/frames
```

(печатает недостающие / подозрительно маленькие кадры; код возврата 0 — всё на месте).

## 6. Монтаж итогового видео

Одной строкой (подходит и для PowerShell, и для bash):

```bash
python stage2_logistics/post/compose2.py --frames stage2_logistics/out/frames --out stage2_logistics/deliverables/demo_full_cycle_1080p.mp4 --compact stage2_logistics/deliverables/demo_full_cycle_1080p_compact.mp4 --verify
```

`compose2.py` сначала проверяет, что отрендерены **все** кадры монтажа и каждый PNG читается целиком (иначе
останавливается и перечисляет проблемные кадры), затем в один проход ffmpeg:

* склеивает новые кадры и отрезки видео этапа 1 по `post/edl.json` (обрезка по номерам кадров, без повторного
  рендера и без наложения титров поверх вставок — на них уже есть свои титры этапа 1);
* накладывает титул, подписи, метку «ДЕМО · СИМУЛЯЦИЯ · УСКОРЕНО» и финальную карточку со схемой потока
  (`post/storyboard2.json`);
* синтезирует звук на весь ролик (фон цеха, сервоприводы, пневматика, прихватки, дуга, маркер, сканер, AGV);
* `--verify` сравнивает кадры на стыках с исходным видео этапа 1 (PSNR) — стыки должны попасть точно;
* в конце сверяет число кадров результата с монтажным листом.

Время монтажа — несколько минут на CPU.

## 7. Если меняли хореографию, камеры или подписи

| Что поменяли | Что запустить |
|---|---|
| `layout2.py`, `plan2.py` (хореография) | `python stage2_logistics/plan2.py` (пересчёт, ≈ 3 мин), затем `python stage2_logistics/edl.py`, перерендер затронутых планов |
| `cameras2.py` (планы, длительности) | `python stage2_logistics/edl.py` (пишет `post/edl.json` и `post/storyboard2.json`), перерендер изменённых планов |
| тексты подписей / карточки | `post/storyboard2.py` (тексты в начале файла), затем `python stage2_logistics/post/storyboard2.py` и только монтаж (п. 6) |

`compose2.py` сам остановится, если `storyboard2.json` сделан для другого монтажного листа.

## 8. Типичные проблемы

* **`llvmpipe` в строке renderer / очень медленно.** EEVEE не видит видеокарту. Windows: обновите драйвер и
  запускайте из обычной сессии пользователя (не из службы/RDP без GPU). Linux без монитора: нужен драйвер с EGL
  (NVIDIA ≥ 470) — Blender 4.5 сам использует EGL в фоновом режиме; `xvfb-run` НЕ использовать (он даёт
  программный OpenGL).
* **EEVEE не запускается в фоне на вашей системе.** Запасной путь — Cycles на GPU:
  `--engine CYCLES --gpu OPTIX --samples 64` (или `CUDA`, `HIP`, `METAL`, `ONEAPI`). Внимание: освещение Cycles
  отличается от EEVEE, и новые планы будут выглядеть иначе, чем вставки сварки этапа 1.
* **Не хватает видеопамяти.** Уменьшите `--samples`, закройте другие программы.
* **`ModuleNotFoundError: PIL` в варианте Б.** Поставьте Pillow во встроенный Python Blender (п. 2).
* **Нужен сам .blend.** `python stage2_logistics/build2.py --save stage2_logistics/out/stage2.blend` — сцену можно
  открыть в Blender 4.5 и посмотреть/подвигать камеры (камера каждого плана — объект `S2_..` с маркером на таймлайне).

## 9. Контрольные числа

* Сцена этапа 2: 4595 кадров; сварка этапа 1 встроена со сдвигом `WELD_OFFSET` = 1840 (печатается при сборке).
* Монтаж: 15 новых планов + 2 вставки этапа 1, ≈ 89,7 с (2152 кадра, из них 1402 новых).
* Проверки перед рендером (на CPU, без Blender-рендера): `python stage2_logistics/tests/t_continuity2.py`,
  `python stage2_logistics/tests/t_collision2.py`, `python stage2_logistics/tests/t_cameras2.py`,
  `python stage2_logistics/tests/t_build2.py` (все печатают `RESULT: OK`).
* Как выглядят планы — `stage2_logistics/deliverables/preview_shots_640x360.jpg` (середина каждого нового плана,
  EEVEE на CPU) и эскизы компоновки `deliverables/layout_top.png`, `deliverables/layout_34.png`.
