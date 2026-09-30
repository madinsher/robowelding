# Инструкция: итоговый рендер и монтаж двух версий видео этапа 2 на машине с видеокартой

Готовятся **две версии** ролика одинаковой длины (1920×1080, 24 fps, ≈ 90 с, звук) и компактные копии:

| Версия | Язык титров и надписей в сцене | Логотип | Сварочная часть | Кадров рендерить | Итоговые файлы (`stage2_logistics/deliverables/`) |
|---|---|---|---|---|---|
| `ru` | русский | ПИГРУПП | **вставка готового видео этапа 1**, не рендерится | 1402 | `demo_full_cycle_ru_pigrupp_1080p.mp4`, `..._compact.mp4` |
| `en` | английский | ATOMIX | рендерится заново из сцены этапа 2 теми же камерами этапа 1 | 2152 | `demo_full_cycle_en_atomix_1080p.mp4`, `..._compact.mp4` |

Почему в английской версии сварка рендерится заново: в готовом видео этапа 1 вшиты русские титры и подписи. Сцена
этапа 2 содержит всю хореографию сварки этапа 1 (со сдвигом `WELD_OFFSET` = 1840 кадров), поэтому те же планы
получаются из неё кадр в кадр (сравнение — `stage2_logistics/out/tests/s1_compare.jpg` после `tests/t_build2.py`),
только с английскими подписями и с участком логистики на заднем плане.

Логотип стоит: крупно на титуле, небольшим знаком в левом верхнем углу всего ролика (в русской версии — и поверх
вставок этапа 1), на финальной карточке, а в самой сцене — на экранах HMI участка и поста контроля, на двери
шкафа управления и на табличке стеллажа. Все команды принимают `--edition ru` или `--edition en` (по умолчанию `ru`).

Схема работы:

```
build2.py --edition ru --render    ->  out/frames_ru/frame_NNNN.png   (NNNN = кадр сцены этапа 2)
build2.py --edition en --render    ->  out/frames_en/frame_NNNN.png
compose2.py --edition ru --final   ->  монтаж по post/edl_ru.json + титры post/storyboard2_ru.json + звук
                                       + отрезки готового видео этапа 1 (demo_welding_cell_1080p.mp4)
compose2.py --edition en --final   ->  монтаж по post/edl_en.json + post/storyboard2_en.json + звук
```

## 1. Что нужно на машине

* Видеокарта с поддержкой OpenGL 4.3 (NVIDIA / AMD / Intel Arc), свежий драйвер, 6+ ГБ видеопамяти.
* Git, ffmpeg в `PATH` (`ffmpeg -version` должен работать).
* **Python 3.11** (именно 3.11 — под него собран модуль `bpy` 4.5) **или** Blender 4.5 LTS (вариант Б ниже).
* ~25 ГБ свободного места (PNG 1080p ≈ 4–6 МБ на кадр: 1402 кадра `ru` + 2152 кадра `en`).

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

Вариант Б — приложение Blender 4.5 LTS. В его встроенный Python нужно поставить Pillow (им рисуются надписи и
маркировка в сцене):

```bash
"<папка Blender>/4.5/python/bin/python3.11" -m pip install pillow        # Linux/macOS
"<папка Blender>\4.5\python\bin\python.exe" -m pip install pillow         # Windows
```

Дальше в командах **рендера** вместо `python stage2_logistics/build2.py ARGS` пишите
`blender -b --python stage2_logistics/build2.py -- ARGS` (аргументы скрипта — после `--`). Монтаж и проверки без
Blender-рендера (`post/compose2.py`, `post/storyboard2.py`, `tests/t_post2.py`) запускайте встроенным Python
Blender'а — в нём есть numpy, а Pillow вы поставили выше: `"<папка Blender>/4.5/python/bin/python3.11"
stage2_logistics/post/compose2.py ...` (Windows: `"<папка Blender>\4.5\python\bin\python.exe" ...`).
`edl.py` нужен только после правки хореографии или камер (монтажные листы обеих версий уже лежат в репозитории); в
варианте Б его запускают так: `blender -b --python stage2_logistics/edl.py -- --edition all`.

Шрифты и логотипы лежат в репозитории (`stage2_logistics/assets/fonts`, `stage2_logistics/assets/logos`),
ставить ничего не нужно.

## 3. Пробный кадр и проверка, что считает видеокарта

```bash
python stage2_logistics/build2.py --edition ru --shot S2_15_scan --res 1920 1080 --samples 16
python stage2_logistics/build2.py --edition en --shot S2_15_scan --res 1920 1080 --samples 16
```

(`--shot ИМЯ` без `--render` рендерит середину плана: здесь экран поста контроля с надписями на языке версии и
логотипом. Имена планов печатает `python stage2_logistics/edl.py`; сварочные планы английской версии называются
`S1_...`, например `--edition en --shot S1_C3a_laser`; любой кадр сцены — `--still N`.)

* Первый запуск дольше (≈ 3–5 мин на CPU): строится траектория сварки этапа 1 (кэш `demo_video/out/anim_cache_*.npz`).
  Хореография этапа 2 уже посчитана и лежит в репозитории (`stage2_logistics/data/plan2_*.npz`, берётся, если код
  хореографии не менялся). Следующие запуски берут кэш, сборка сцены ≈ 2–3 мин.
* Кадр появится в `stage2_logistics/out/stills_ru/` (или `stills_en/`) как `frame_NNNN.png`.
* В логе есть строка `[build2] edition ru: lang ru, brand pigrupp, ...` (или `en ... atomix`) и строка
  `[render] engine BLENDER_EEVEE_NEXT, OpenGL/Vulkan renderer: ...` — там должно быть имя вашей видеокарты
  (например `NVIDIA GeForce RTX ...`). Если там `llvmpipe` / `Software` — рендер идёт на процессоре, см. раздел 8.
* Строка `[render] frame NNNN ... NN.Ns` — время кадра. Первый кадр включает компиляцию шейдеров (до 1–2 мин),
  дальше на современной видеокарте ожидается ≈ 2–6 с на кадр 1080p.

## 4. (Необязательно) быстрый черновик ролика

```bash
python stage2_logistics/build2.py --edition en --preview                 # каждый 6-й кадр монтажа, 640×360
python stage2_logistics/post/compose2.py --edition en --preview          # -> stage2_logistics/out/preview_en.mp4
```

То же с `--edition ru`. Черновик проверяет монтаж, титры, логотипы и звук до долгого рендера (пропущенные кадры
«держатся» предыдущими).

## 5. Финальный рендер новых кадров

```bash
python stage2_logistics/build2.py --edition ru --render --samples 16      # 1402 кадра -> out/frames_ru
python stage2_logistics/build2.py --edition en --render --samples 16      # 2152 кадра -> out/frames_en
```

* Версии независимы: можно считать в любом порядке или на разных машинах. Кадры сцены у версий разные (надписи и
  логотипы в сцене), поэтому у каждой своя папка.
* Рендер **возобновляемый**: при повторном запуске готовые кадры пропускаются — можно прервать (Ctrl+C) и продолжить
  той же командой.
* Настройки рендера — как в этапе 1 (EEVEE, AgX, тени от ключевого света и дуги), чтобы новые планы по картинке
  совпадали со сваркой этапа 1. `--samples 16` убирает шум сглаживания (в этапе 1 было 8 из-за CPU); можно 24–32,
  если время позволяет (для `en` это заметнее: сварочные планы в нём тоже новые).
* `--quality high` включает тени от всех источников и screen-space отражения — картинка богаче, но заметнее
  отличается от вставок этапа 1 в русской версии. По умолчанию не использовать.
* Несколько машин: разделите работу — `--worker 0/2` на одной и `--worker 1/2` на другой (с тем же `--edition`),
  затем сложите PNG в одну папку. На одной машине запускайте один процесс: EEVEE и так загружает видеокарту целиком.
* Папка кадров помечена хэшем хореографии и версией (`out/frames_<версия>/.plan_id`). Если запустить `--render` в
  папку с кадрами другой версии или старой хореографии, `build2.py` откажется смешивать кадры — перенесите старые
  или укажите другую `--outdir`.
* Только один план (например, после правки камеры): `--edition ru --shot S2_08_load --render` (удалите старые кадры
  этого плана, иначе они будут пропущены). Имена планов — `stage2_logistics/cameras2.py`.

Проверка комплектности:

```bash
python stage2_logistics/edl.py --edition ru --check-frames stage2_logistics/out/frames_ru
python stage2_logistics/edl.py --edition en --check-frames stage2_logistics/out/frames_en
```

(печатает недостающие / подозрительно маленькие кадры; код возврата 0 — всё на месте).

## 6. Монтаж итоговых видео

```bash
python stage2_logistics/post/compose2.py --edition ru --final
python stage2_logistics/post/compose2.py --edition en --final
```

`--final` берёт кадры из `out/frames_<версия>` и пишет `deliverables/demo_full_cycle_<версия>_<логотип>_1080p.mp4` и
`..._compact.mp4`. `compose2.py` сначала проверяет, что отрендерены **все** кадры монтажа и каждый PNG читается
целиком (иначе останавливается и перечисляет проблемные кадры), затем в один проход ffmpeg:

* склеивает кадры по монтажному листу версии (`post/edl_ru.json` / `post/edl_en.json`); в русской версии вставляет
  отрезки видео этапа 1 с точностью до кадра, без повторного рендера и без своих титров поверх них (на них уже есть
  титры этапа 1) — поверх идёт только логотип в углу;
* накладывает титул с логотипом, подписи, метку «ДЕМО · СИМУЛЯЦИЯ · УСКОРЕНО» / «DEMO · SIMULATION · SPED UP»,
  логотип в углу и финальную карточку со схемой потока и логотипом (`post/storyboard2_<версия>.json`); в
  английской версии подписи сварочной части этапа 1 идут в переводе в те же моменты, что и в видео этапа 1;
* синтезирует звук на весь ролик (одинаковый в обеих версиях: фон цеха, сервоприводы, пневматика, прихватки, дуга,
  маркер, сканер, AGV);
* для `ru` проверяет стыки с исходным видео этапа 1 (PSNR) — стыки должны попасть точно;
* в конце сверяет число кадров результата с монтажным листом.

Время монтажа — несколько минут на CPU для каждой версии.

## 7. Если меняли хореографию, камеры, тексты или логотипы

| Что поменяли | Что запустить |
|---|---|
| `layout2.py`, `plan2.py` (хореография) | `python stage2_logistics/plan2.py` (пересчёт, ≈ 3 мин), затем `python stage2_logistics/edl.py`, перерендер затронутых планов обеих версий |
| `cameras2.py` (планы, длительности) | `python stage2_logistics/edl.py` (пишет `post/edl_<версия>.json` и `post/storyboard2_<версия>.json` обеих версий), перерендер изменённых планов |
| тексты титров / подписей / карточки | `post/storyboard2.py` (русские и английские тексты в начале файла), затем `python stage2_logistics/post/storyboard2.py` и только монтаж (п. 6) |
| надписи в сцене (экраны, таблички) | английские тексты — `stage2_logistics/i18n2.py`; проверка `python stage2_logistics/tests/t_i18n2.py`; перерендер планов, где надпись видна |
| логотип | заменить `assets/logos/<бренд>.png` (прозрачный фон; из JPG на белом фоне — `assets/logos/clean_logo.py`); перерендер (логотип есть и в сцене) и монтаж |
| какая версия с каким языком / логотипом | `stage2_logistics/editions.py` |

`compose2.py` сам остановится, если титры сделаны для другой версии или другого монтажного листа.

## 8. Типичные проблемы

* **`llvmpipe` в строке renderer / очень медленно.** EEVEE не видит видеокарту. Windows: обновите драйвер и
  запускайте из обычной сессии пользователя (не из службы/RDP без GPU). Linux без монитора: нужен драйвер с EGL
  (NVIDIA ≥ 470) — Blender 4.5 сам использует EGL в фоновом режиме; `xvfb-run` НЕ использовать (он даёт
  программный OpenGL).
* **EEVEE не запускается в фоне на вашей системе.** Запасной путь — Cycles на GPU:
  `--engine CYCLES --gpu OPTIX --samples 64` (или `CUDA`, `HIP`, `METAL`, `ONEAPI`). Внимание: освещение Cycles
  отличается от EEVEE, и в русской версии новые планы будут выглядеть иначе, чем вставки сварки этапа 1.
* **Не хватает видеопамяти.** Уменьшите `--samples`, закройте другие программы.
* **`ModuleNotFoundError: PIL` в варианте Б.** Поставьте Pillow во встроенный Python Blender (п. 2).
* **Рендер своим скриптом.** Рендерите только через `build2.py`: перед рендером он «запекает» частицы искр
  (`vfx.bake_particles`); без этого при переходах между далёкими кадрами роботы могут встать не в свои позы.
* **Нужен сам .blend.** `python stage2_logistics/build2.py --edition en --save stage2_logistics/out/stage2_en.blend` —
  сцену можно открыть в Blender 4.5 и посмотреть/подвигать камеры (камера каждого плана — объект `S2_..` или
  `S1_..` с маркером на таймлайне).

## 9. Контрольные числа

* Сцена этапа 2: 4595 кадров; сварка этапа 1 встроена со сдвигом `WELD_OFFSET` = 1840 (печатается при сборке).
* Монтаж обеих версий: 2152 кадра ≈ 89,7 с. `ru`: 15 новых планов + 2 вставки видео этапа 1 (1402 кадра рендерить).
  `en`: те же 15 планов + 9 сварочных планов этапа 1 из сцены этапа 2 (2152 кадра рендерить).
* Проверки перед рендером (на CPU, без финального рендера): `python stage2_logistics/tests/t_continuity2.py`,
  `t_collision2.py`, `t_cameras2.py`, `t_build2.py --edition ru`, `t_build2.py --edition en`, `t_i18n2.py`,
  `t_post2.py` (все печатают `RESULT: OK` или число пройденных проверок).
* Как выглядят планы — `stage2_logistics/deliverables/preview_shots_640x360.jpg` (середина каждого нового плана,
  EEVEE на CPU), сравнение версий — `deliverables/editions_preview.jpg`, эскизы компоновки
  `deliverables/layout_top.png`, `deliverables/layout_34.png`.
