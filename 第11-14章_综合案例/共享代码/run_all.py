"""按第11章到第14章的顺序重新生成全部Python结果。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

SCRIPTS = [
    ROOT / "第11章/代码/ch11_research_design.py",
    ROOT / "第12章/代码/ch12_data_preparation.py",
    ROOT / "第13章/代码/ch13_estimation.py",
    ROOT / "第14章/代码/ch14_reporting.py",
]

environment = os.environ.copy()

for script in SCRIPTS:
    print(f"\n>>> 运行 {script.relative_to(ROOT)}")
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        env=environment,
    )

print("\n全部章节已重新生成。")
