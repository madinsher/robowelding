# post/ — пост-продакшн демо-видео

Превращает последовательность отрендеренных кадров (`out/frames/frame_%04d.png`, 24 fps,
кадры 1…1008) в финальный ролик `demo_final.mp4` (1920×1080, H.264 crf 18, AAC) с русскими
титрами и синтезированным звуком. Никаких внешних медиафайлов: титры рисуются PIL шрифтами
DejaVu, звук генерируется numpy.

```
post/storyboard.json   раскадровка: титульный кадр, подписи, финальный кадр, шрифты, звук
post/overlays.py       рендер титров в RGBA-PNG (PIL)
post/audio.py          синтез саундтрека (48 кГц стерео WAV, пик −3 dBFS)
post/compose.py        CLI: кадры + титры + звук -> mp4 (ffmpeg)
tests/t_post.py        смоук-тест на синтетических кадрах
```

## Запуск

```bash
cd demo_video
python3 post/compose.py --frames out/frames --out out/demo_final.mp4          # полный ролик
python3 post/compose.py --frames out/frames --out out/demo_final.mp4 --no-audio
python3 post/compose.py --frames out/preview_frames --out out/preview.mp4 --preview
python3 post/audio.py --out out/soundtrack.wav                                # только звук
python3 tests/t_post.py                                                       # тест (≈1.5 мин)
```

Параметры `compose.py`:

| ключ | назначение |
|---|---|
| `--frames DIR` | папка с `frame_NNNN.png` (номер = кадр раскадровки) |
| `--out FILE` | выходной mp4 |
| `--storyboard FILE` | раскадровка (по умолчанию `post/storyboard.json`) |
| `--no-audio` | без звуковой дорожки |
| `--preview` | быстрый пресет x264; допустимы разреженные кадры (каждый 8-й) и низкое разрешение |
| `--keep-temp` | сохранить PNG титров и WAV рядом с выходом (`<out>_tmp/`) |

Кадры могут быть **разреженными** (например, каждый 8-й из `build.py --preview`) — тогда они
трактуются как последовательность 3 fps и растягиваются до 24 fps, титры и звук остаются
синхронными с раскадровкой. Можно подать и **фрагмент** (например, кадры 930…1008): время
отсчитывается от номера первого кадра, звук обрезается соответственно. Кадры любого размера
масштабируются до 1920×1080.

Конвейер целиком на ffmpeg: кадры → `scale` → `fps=24` → цепочка `overlay` с
`enable='between(t,a,b)'` и `fade … alpha=1` (плавное появление/исчезновение ~12 кадров) →
`yuv420p`, libx264 crf 18, `+faststart`; звук → AAC 192 kbps.

## Как менять подписи

Всё в `post/storyboard.json` (времена — в номерах кадров при 24 fps, включительно):

* `title` — титульная плашка (тёмный градиент внизу кадра, 3D-сцена остаётся видимой):
  `heading`, `sub`, `start`, `end`.
* `captions` — нижние титры: `tag` (короткая метка в оранжевом чипе, автоматически в верхнем
  регистре), `text` (переносится по словам, ширина ≤1160 px), `start`, `end`. Добавляйте или
  удаляйте элементы списка; интервалы не должны пересекаться.
* `end_card` — финальный кадр (затемнение до ~45 %): `heading`, `lines` (маркированный список).
* `corner_label` — метка в правом верхнем углу (60 % непрозрачности, весь ролик).
* `footer` — мелкая строка внизу финального кадра.
* `font`, `font_bold` — пути к TTF (нужна кириллица; по умолчанию DejaVu Sans).
* `style` (необязательно) — переопределение цветов/отступов из `overlays.STYLE`, например
  `"style": {"accent": [255, 106, 19, 255], "margin_x": 96}`.

Быстрая проверка вёрстки без ffmpeg:

```python
from post import overlays; from PIL import Image; import json
sb = json.load(open("post/storyboard.json"))
ovs = overlays.render_all(sb, "out/ov_tmp")
overlays.preview_frame(Image.open("out/int/frame_0300.png"), ovs, 300).save("out/check.png")
```

## Звук

`audio.py` собирает дорожку из трёх слоёв, интервалы берутся из подписей раскадровки:

* **фон цеха** — фильтрованный коричневый + розовый шум, −32 dBFS RMS;
* **сервоприводы** — мягкая пила 200–400 Гц с разгоном/торможением во время подписей,
  чьи `tag` перечислены в `audio.servo_tags` (±`servo_pad_s` с);
* **дуга MIG** — пачки белого шума со случайной частотой 30–120 Гц, полоса 1–6 кГц, фон
  50/100 Гц, плавные 0,3 с на границах, щелчок зажигания — во время подписей, чей `tag`
  начинается с `audio.weld_prefix` («ШОВ»).

Уровни (`*_db`), `seed` и списки меток настраиваются в секции `audio` раскадровки.
Итог нормализуется к пику `peak_db` (−3 dBFS).

## Тест

`python3 tests/t_post.py` генерирует синтетические кадры (`out/post_test_frames_sparse/` —
каждый 8-й кадр 960×540, `out/post_test_frames_excerpt/` — кадры 930…1008 в 1080p), собирает
`out/post_test_preview.mp4`, `out/post_test_excerpt.mp4`, `out/post_test_excerpt_noaudio.mp4`
и вырезает контрольные стоп-кадры `out/post_check_*.png` (титул, подписи, финальный кадр).
`--full` дополнительно прогоняет все 1008 кадров.
