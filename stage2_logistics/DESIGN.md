# Этап 2 — техническая спецификация модулей (контракт)

Этот документ — рабочий контракт между модулями `stage2_logistics/`. Координаты и размеры — только из
`layout2.py` (единый источник), кинематика — `kin.py`. Этап 1 (`demo_video/`) **не редактируется**: его
модули импортируются через `tools.py` (`from cell import layout, geom as G, materials, spool, ...`).

## 0. Общая идея

* Сцена этапа 2 = сцена этапа 1 (ячейка сварки, позиционер, сварочный робот, цех) + зона логистики на стороне
  загрузки (−X): погрузчик на линейном треке, стенд сборки/прихватки с малым роботом, кассета заготовок, буфер
  труб, выходной рольганг с паллетами-спутниками, арка маркировки и контроля, ступенчатый стеллаж готовых, AGV.
* Хореография сварки этапа 1 (1104 кадра) встраивается в таймлайн этапа 2 со сдвигом `WELD_OFFSET`
  (кадр этапа 2 = `WELD_OFFSET` + кадр этапа 1). **Сварка заново не рендерится**: монтаж (`edl.py`,
  `post/compose2.py`) вставляет отрезки готового видео этапа 1 между новыми сценами.
* Все движения считаются в numpy (`plan2.py`), затем ключуются в Blender (`animation2.py`) через функции-сеттеры
  модулей оборудования (ниже).

## 1. Системы координат

* Мир этапа 1: метры, Z вверх, позиционер в начале координат, трек сварочного робота вдоль Y при x = 2.0,
  ограждение ячейки x ∈ [−2.4, 4.2], y ∈ [−3.4, 3.4], проём загрузки со световой завесой на −X при |y| < 1.2.
* **Кадр спула** («стоячая» поза): начало — центр задней (привалочной) плоскости фланца, локальная Z вверх,
  отвод вверх, труба горизонтально вдоль локальной +X (как на позиционере при наклоне 0). Кадр задаётся
  `(origin, x_dir)` → `kin.planar_frame(origin, x_dir)`. Геометрия в локальных координатах: шов A — окружность
  радиуса `PIPE_R` при z = 0.10 (ось Z); шов B — окружность при x = 0.381 (ось X), центр (0.381, 0, 0.481);
  труба x ∈ [0.381, 0.981], ось на z = 0.481; верх изделия z = 0.618; фланец Ø0.405, толщина 0.028.
* **Детали** (фланец, отвод, труба) — отдельные корневые empty (`part_flange`, `part_elbow`, `part_pipe`),
  чей кадр **совпадает с кадром спула в собранном виде** (без смещений). Деталь «на своём месте в спуле с кадром F»
  имеет мировую позу F. Мировые позы деталей ключуются покадрово (location + quaternion).
* **Хват** (`layout2.GRASP[name]`): точка на оси захватываемого цилиндра, z — направление подхода (вниз),
  x — ось цилиндра. TCP захвата совпадает с кадром хвата: `T_tcp = T_part @ frame(GRASP)` (или повёрнутым на 180°
  вокруг z — захват симметричен).
* **TCP захвата** = `tool0 @ Tz(GRIP_TCP_Z)`, оси = оси `tool0`: z — от фланца робота к детали, x — ось шарниров губок
  (ось трубы), y — направление сжатия губок.

## 2. Правила для всех модулей

* Каждый `build()` создаёт свою коллекцию (`bpy.data.collections.new(NAME)` → `scene.collection.children.link`)
  и кладёт туда только свои объекты; возвращает `dict`. Имена объектов — с префиксом модуля (таблица ниже),
  чтобы не конфликтовать с этапом 1 и друг с другом.
* Материалы — `materials.get(...)` этапа 1: `abb_orange abb_grey dark_metal black_plastic brass copper
  machined_steel cast_iron galvanized painted(color=, roughness=, coat=) safety_yellow safety_red rubber
  cable_black glass_dark emissive(color=, strength=) steel_pipe bevel_steel concrete fence_mesh(...)` и др.
  Цветовая гамма как на кадрах этапа 1: машины — синий `#1F4E8C`/`#26558F`, рамы — `#2B2F36`, ограждения —
  жёлтые стойки + тёмная сетка, опасные зоны — жёлто-чёрная штриховка, роботы — ABB-оранжевый.
  Новые материалы — создавать с префиксом модуля в имени (`bpy.data.materials.new("conv_...")`).
* Бюджет геометрии: ≤ 150 тыс. треугольников на модуль (окружение этапа 1 ~93 тыс. полигонов).
* Анимация — только через сеттеры `set_*(handle, value, frame=None)`: ставят значение и, если `frame` задан,
  вставляют ключ. Интерполяцию (LINEAR/CONSTANT) выставляет планировщик. Сеттер **не** трогает другие каналы.
* Анимируемые подвижные узлы — отдельные объекты/empty с понятным локальным каналом (например, `location[0]`
  каретки, `rotation_euler[0]` рычага прижима). Никаких драйверов на кадр/время, никакой физики.
* Никаких отладочных `print` в модулях; тесты — в `stage2_logistics/tests/`, результаты — в
  `stage2_logistics/out/` (в .gitignore).
* Каждый модуль экспортирует `obstacles() -> list[dict(name, center=(x,y,z), size=(sx,sy,sz), yaw=0.0)]` —
  грубые боксы статической геометрии в мировых координатах (для численной проверки коллизий `t_collision2.py`,
  без bpy). Боксы должны **покрывать** геометрию (консервативно), но не «съедать» зоны хвата.
* Blender 4.5: `bpy.ops.wm.stl_import`, `shade_smooth_by_angle`, EEVEE = `BLENDER_EEVEE_NEXT`, свет — `energy`.
* Хелперы геометрии — `cell.geom` (`box`, `cylinder`, `revolve`, `torus_sweep`, `empty`, `set_parent`, `M`),
  ограждение/стойки — можно вызывать приватные функции `cell.environment` (`_B`, `_mats`, `_fence_bay`, `_post`,
  `_box`, `_rod`, `_pipe`, `_bar`, `_prism`, `_rev`) — так новое ограждение будет выглядеть как в этапе 1.

| Модуль | Префикс | Коллекция |
|---|---|---|
| `parts.py` | `part_` | `Parts2` |
| `handler_robot.py` | `hnd_` | `Handler` |
| `gripper.py` | `grp_` | (в коллекции робота) |
| `tack_robot.py` | `tack_` | `TackRobot` |
| `assembly_station.py` | `stn_` | `AssemblyStation` |
| `cassette.py` | `kit_` | `KitCassette` |
| `conveyor.py` | `conv_` | `OutputConveyor` |
| `marking_qc.py` | `qc_` | `MarkingQC` |
| `storage.py` | `sto_` | `Storage` |
| `environment2.py` | `env2_` | `Environment2` |

## 3. API модулей

### parts.py
* `split(sp) -> dict(flange, elbow, pipe, roots, tacks)` — принимает словарь `spool.build(name="spool")` этапа 1
  (уже установленный на позиционер и анимированный `animation.build`), создаёт empty `part_flange`,
  `part_elbow`, `part_pipe` в текущей мировой позе корня спула и переподчиняет им объекты с сохранением мировой
  позы: фланец ← `spool_flange, spool_flange_face, spool_hole*, spool_bead_A, spool_tint_A`; отвод ←
  `spool_elbow, spool_elbow_bevel_A/B, spool_mark_elbow, spool_bead_B, spool_tint_B`; труба ← `spool_pipe,
  spool_mark_pipe`. Корень спула этапа 1 затем удаляется/отвязывается. Имена `spool_bead_*`, `spool_tint_*`
  сохраняются (на них ссылается анимация этапа 1). `tacks = dict(A=[3 obj], B=[3 obj])` — короткие валики
  прихваток (длина `TACK_LEN`) в точках `TACKS_A/TACKS_B`, дочерние к фланцу (A) и отводу (B).
* `set_pose(root, T, frame=None)` — мировая поза корня детали (numpy 4×4) → `location` + `rotation_quaternion`.
* `set_tack(obj, s, hot=0.0, frame=None)` — `s` ∈ [0,1] рост прихватки (0 — невидима), `hot` ∈ [0,1] свечение.
* `build_kit(main, name) -> dict(flange, elbow, pipe)` — ещё один комплект заготовок (для следующего цикла и
  «запаса» в кассете): linked-duplicates мешей основных деталей, те же корни/кадры, без валиков/прихваток.
* `build_finished(main, name) -> root` — готовый сваренный спул (один корень, кадр спула): швы A и B полностью
  видимы, окалина (tint) видна, без свечения; linked-duplicates.

### handler_robot.py
* `build_arm(prefix, scale, base_world, collection, colours=None) -> dict(arm, base, joints, tool0, links)` —
  универсальная сборка IRB 4600 × `scale` (меши звеньев масштабированы, происхождения суставов — как
  `kin.scaled_arm(scale)`), empty суставов `J1..J6` анимируются `rotation_axis_angle[0]` как в этапе 1
  (`robot_build.set_q` совместим). `base_world` — numpy 4×4. Проверка: FK bpy совпадает с `kin.scaled_arm`.
* `build(collection=None) -> dict(collection, arm, carriage, base, joints, tool0, links, gripper)` — погрузчик:
  напольный трек вдоль X (`HANDLER_TRACK_BED`, y = `HANDLER_TRACK_Y`): станина, рельсы, рейка, кабель-канал с цепью,
  упоры; каретка (empty, анимируется `location[0]` = x оси J1) с корпусом и плитой на высоте `HANDLER_TRACK_TOP_Z`;
  робот × `HANDLER_SCALE`, yaw `HANDLER_YAW`; пакет кабелей/шлангов от основания до запястья (упрощённо, по образцу
  `robot_build._build_hose`); захват `gripper.build(tool0)`.
* `set_track(rob, x, frame=None)`, `set_q(rob, q, frame=None)`.

### gripper.py
* `build(tool0, collection, name="grp") -> dict(root, tcp, jaws, parts)` — клещевой захват: переходная плита,
  поворотный модуль, корпус с пневмоцилиндром, две губки на шарнирах (ось ∥ tool X) с V-призмами. При `s = 0`
  призмы касаются цилиндра радиуса `PIPE_R`, ось которого проходит через TCP вдоль tool X; при `s = 1` губки
  раскрыты так, что пропускают цилиндр Ø0.36 м при подходе сверху. `tcp` — empty в `Tz(GRIP_TCP_Z)`.
  Габарит: вдоль tool Z от 0 до `GRIP_TCP_Z + PIPE_R + 0.06`, вдоль tool Y ±0.32 (раскрыт), вдоль tool X ±0.13.
* `set_open(g, s, frame=None)`.

### tack_robot.py
* `build(collection=None) -> dict(collection, arm, base, joints, tool0, tcp, arc, torch)` — малый робот
  (IRB 4600 × `TACK_SCALE` через `handler_robot.build_arm`) на пьедестале `TACK_BASE`, высота `TACK_BASE_Z`,
  yaw `TACK_YAW`; горелка MIG этапа 1 (`robot_build._build_torch`, объекты переименовать в `tack_torch_*`),
  `tcp` = `tool0 @ robot_build.tool_transform()`; `arc` — empty в TCP (его +Z — в изделие, как `arc_point` этапа 1)
  со свойством `arc_on`; подающий механизм на плече, упрощённый шланг-пакет; небольшой шкаф/источник рядом.
* `set_q(rob, q, frame=None)`; `set_arc(rob, on, frame=None)` (свойство `arc_on`, CONSTANT).

### assembly_station.py
* `build(collection=None) -> dict(collection, clamps, lamp, table, frame)` — сварной стол `STATION_TABLE`, плита,
  центрирующее посадочное место фланца с 3 штифтами, V-блоки под отводом (лок. x ≈ 0.30) и под трубой (лок. x ≈ 0.86),
  упор торца трубы (задаёт зазор), 4 пневмоприжима (`STATION_CLAMPS`: рычаг + цилиндр + шток + пята), датчик зазора
  на кронштейне с лампой, пневмоостров со шлангами, пульт. `frame` — кадр спула на стенде (numpy 4×4).
* **Свободные коридоры**: над точками хвата `GRASP[flange|elbow|pipe|spool]` в кадре стенда — вертикальные
  цилиндры радиусом 0.20 м от детали до z = 3.0 — ничего не ставить (в раскрытом состоянии прижимов тоже).
  Сторона +Y локального кадра (мировая +X) у швов A и B — доступ горелки прихваточного робота.
* `set_clamp(st, name, s, frame=None)` — s = 0 раскрыт, 1 зажат. `set_lamp(st, state, frame=None)` — 0 выкл, 1 зелёный.

### cassette.py
* `build(collection=None) -> dict(collection, pallet, nests, pipe_buffer)` — паллета `KIT_PALLET` со стальной
  кассетой: 2 гнезда фланцев (лёжа, привалочной плоскостью вниз, 3 штифта) в кадрах `KIT_FLANGES`, 2 гнезда
  отводов (стакан под нижний торец + V-опора под верхний конец) в кадрах `KIT_ELBOWS`; буфер труб `PIPE_BUFFER`:
  стойка с V-ложементами под 3 трубы в кадрах `pipe_buffer_frame(i)`, датчики наличия. Сами детали не строятся.

### conveyor.py
* `build(collection=None, n_carriers=2) -> dict(collection, carriers, rollers)` — приводной рольганг `CONVEYOR`
  (рама, ноги, ролики вдоль Y, привод, боковые направляющие, упоры, фотодатчики), паллеты-спутники `CARRIER`:
  плита, посадочное кольцо фланца с 3 штифтами (кадр спула), V-стойка под трубой (лок. x ≈ 0.86), бирка RFID.
  Корень паллеты = кадр спула на ней (origin на `seat_z`, x_dir = +Y мира).
* `set_carrier(conv, i, x, frame=None)` — x центра фланца (y = `CONV_SPOOL_Y`, z = `seat_z`).
* `set_rollers(conv, travel_m, frame=None)` — поворот всех роликов на `travel_m / roller_r`.

### marking_qc.py
* `build(collection=None) -> dict(collection, root, carriage, marker, beam, scanner, sensor, tower, display)` —
  портал над рольгангом (`QC_ARCH`, всё под одним корнем `qc_root`), каретка вдоль Y, лазерный маркер (головка +
  эмиссивный луч `beam`), сканер профиля шва: `sensor` — empty, чья +Z смотрит вниз на изделие (для
  `vfx.laser_line`), сигнальная колонна (красный/жёлтый/зелёный), экран HMI с надписью.
* `build_mark(pipe_root) -> obj` — декаль маркировки на трубе в `QC_MARK_LOCAL` (DataMatrix + «SPL-02 · WPS-07 ·
  QC OK»), дочерняя к корню трубы; свойство `reveal` 0…1 (проявление слева направо) и `hot` (свечение).
* `set_carriage(qc, y, frame)`, `set_marker(qc, on, frame)`, `set_mark(decal, reveal, hot, frame)`,
  `set_tower(qc, state, frame)` (`"off" | "amber" | "green"`), `set_display(qc, state, frame)`.

### storage.py
* `build(collection=None) -> dict(collection, rack, bays, agv)` — ступенчатый стеллаж `STORAGE` (ярус 0 — низкий
  передний, ярус 1 — высокий задний, места яруса 1 смещены на полшага: V-стойки под трубами яруса 1 стоят в
  промежутках между спулами яруса 0), посадочные места = кольцо фланца + V-стойка под трубой (лок. x ≈ 0.86),
  номера мест; задняя сторона — граница ограждения зоны; `bays[(tier, bay)]` = кадр спула (numpy 4×4).
  AGV (`AGV`): платформа, колёса, лидары, маяк, подъёмный стол; `agv["root"]`.
* `set_agv(st, x, y, yaw, frame=None)`, `set_agv_lift(st, h, frame=None)`, `set_beacon(st, on, frame=None)`.

### environment2.py
* `build(collection=None) -> dict(collection, objects, muting)` — ограждение зоны логистики (стиль этапа 1: жёлтые
  стойки, тёмная сетка) по периметру x ∈ [`LOG_FENCE_X0`, −2.4], y ∈ ±3.4 (стыкуется с углами ограждения ячейки,
  проёмы `LOG_FENCE_OPENINGS` со световыми завесами, дверь `PERSONNEL_DOOR`), разметка пола (жёлтые линии проходов,
  штриховка зоны передачи у проёма ячейки, контуры зон), таблички, шкаф управления зоной/HMI, лампы режима
  «muting» у проёма загрузки ячейки (новые стойки снаружи, x ≈ −2.6, y = ±1.35).
* `build_lighting(scene) -> dict(lights)` — дополнительный свет зоны логистики (`LOG_KEY_LIGHT`, имя
  `env2_light_key` — единственный новый источник с тенями), слабый заполняющий; свет этапа 1 не трогать.
* `set_muting(env, on, frame=None)`.

## 4. Проверки модулей

Каждый модуль имеет `tests/t_<module>.py` на `tests/harness.py`: строит модуль в пустой сцене, проверяет
контракт (ключи словаря, сеттеры ставят ключи, габарит в пределах своей зоны `layout2`), рендерит 2–3 стилла
Cycles 640×360 в `out/tests/`. Запуск: `python3 stage2_logistics/tests/t_<module>.py`.
