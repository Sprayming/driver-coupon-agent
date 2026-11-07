"""pytest 引导：把 backend 目录加入 sys.path，测试里可直接 import store/rag/api。"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
