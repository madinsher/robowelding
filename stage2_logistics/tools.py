"""Единая точка доступа к инструментам этапа 1 (demo_video). Только импорт, без побочных эффектов."""
import os
import sys

DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_video")
if DEMO not in sys.path:
    sys.path.insert(0, DEMO)

from cell import layout, geom, materials, spool, positioner, robot_build, robot_urdf, animation, environment, vfx  # noqa: E402,F401
