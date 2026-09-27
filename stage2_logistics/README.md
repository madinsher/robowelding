# Этап 2 — подача заготовок, обработка и выдача готового изделия

Подпроект расширяет демо-ячейку из `demo_video/` полным материальным потоком: от кассеты с
заготовками до стеллажа готовых спулов. Сейчас в папке только план работ (`PLAN.md`); код появится
по мере выполнения этапов плана.

Подпроект **не копирует** инструменты, а импортирует их из `demo_video/`:

```python
# stage2_logistics/tools.py — единая точка доступа к инструментам этапа 1
import os, sys
DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_video")
sys.path.insert(0, DEMO)
from cell import layout, geom, materials, spool, positioner, robot_build, robot_urdf, animation, environment, vfx  # noqa
```

Что переиспользуется как есть: геометрические помощники (`geom`), библиотека материалов, спул, позиционер,
сборка робота из URDF и численный IK, эффекты дуги, окружение цеха, конвейер рендера (`build.py`,
`render_final.sh`), пост-обработка (`post/compose.py`, `post/audio.py`), RoboDK-скрипт, проверки
(`tests/t_solve.py`, `tests/t_collision.py`, `tests/contact_sheet.py`).
