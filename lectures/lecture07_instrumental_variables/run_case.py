"""
第7讲工具变量案例：运行入口。

用法：
    python3 run_case.py --mode teaching
    python3 run_case.py --mode teaching --stata "/path/to/stata-mp"
    python3 run_case.py --mode reference

teaching 模式运行 Python 分析；若找到 Stata，也会运行 Stata 并对照。
reference 模式重新生成 Python 参考结果并复制到 reference/。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT / "reference"
OUT_PY = ROOT / "output" / "python"
OUT_ST = ROOT / "output" / "stata"

STATA_CANDIDATES = [
    "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se",
]


def run_python() -> None:
    subprocess.run([sys.executable, "scripts/analysis_python.py"],
                   cwd=ROOT, check=True)


def find_stata(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for candidate in STATA_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("stata-mp") or shutil.which("stata-se") or shutil.which("stata")
    return found


def run_stata(executable: str) -> None:
    OUT_ST.mkdir(parents=True, exist_ok=True)
    subprocess.run([executable, "-b", "do", "scripts/analysis_stata.do"],
                   cwd=ROOT, check=True)


def load_key() -> dict:
    return json.loads((REF / "key_numbers.json").read_text(encoding="utf-8"))


def compare_python(key: dict) -> bool:
    import pandas as pd

    ols = pd.read_csv(OUT_PY / "ols_vs_iv.csv")
    fs = pd.read_csv(OUT_PY / "first_stage_diagnostics.csv")
    oid = pd.read_csv(OUT_PY / "overid.csv")
    dwh = pd.read_csv(OUT_PY / "dwh.csv")

    checks = {
        "OLS_RD": (float(ols.loc[ols.variable == "RD", "OLS_coef"].iloc[0]), key["OLS_RD"]),
        "IV_RD": (float(ols.loc[ols.variable == "RD", "IV_coef"].iloc[0]), key["IV_RD"]),
        "first_stage_partial_F": (float(fs.partial_F.iloc[0]), key["first_stage_partial_F"]),
        "overid_p": (float(oid.p_value.iloc[0]), key["overid_p"]),
        "dwh_F": (float(dwh.F.iloc[0]), key["dwh_F"]),
    }
    ok = True
    for name, (got, expected) in checks.items():
        passed = abs(got - expected) <= 1e-4 * max(1.0, abs(expected))
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {got:.6f} (reference {expected:.6f})")
    return ok


def compare_stata(key: dict) -> bool:
    import pandas as pd

    ols = pd.read_csv(OUT_ST / "ols_vs_iv.csv")
    fs = pd.read_csv(OUT_ST / "first_stage_diagnostics.csv")
    oid = pd.read_csv(OUT_ST / "overid.csv")
    dwh = pd.read_csv(OUT_ST / "dwh.csv")

    checks = {
        "OLS_RD": (float(ols.loc[ols.variable == "RD", "OLS_coef"].iloc[0]), key["OLS_RD"]),
        "IV_RD": (float(ols.loc[ols.variable == "RD", "IV_coef"].iloc[0]), key["IV_RD"]),
        "first_stage_partial_F": (float(fs.partial_F.iloc[0]), key["first_stage_partial_F"]),
        "overid_p": (float(oid.p_value.iloc[0]), key["overid_p"]),
        "dwh_F": (float(dwh.F.iloc[0]), key["dwh_F"]),
    }
    ok = True
    for name, (got, expected) in checks.items():
        passed = abs(got - expected) <= 1e-3 * max(1.0, abs(expected))
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {got:.6f} (reference {expected:.6f})")
    return ok


def make_reference() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    for path in OUT_PY.glob("*.csv"):
        shutil.copy2(path, REF / path.name)
    print(f"reference updated: {REF}")


def main() -> int:
    parser = argparse.ArgumentParser(description="第7讲工具变量案例")
    parser.add_argument("--mode", choices=["teaching", "reference"], default="teaching")
    parser.add_argument("--stata", default=None, help="Stata 可执行文件路径")
    parser.add_argument("--no-stata", action="store_true", help="只运行 Python")
    args = parser.parse_args()

    print("=== 运行 Python 分析 ===")
    run_python()

    if args.mode == "reference":
        make_reference()
        return 0

    key = load_key()
    print("\n=== 与参考结果对照（Python） ===")
    ok_py = compare_python(key)

    stata = None if args.no_stata else find_stata(args.stata)
    ok_st = True
    if stata:
        print(f"\n=== 运行 Stata 分析：{stata} ===")
        run_stata(stata)
        print("\n=== 与参考结果对照（Stata） ===")
        ok_st = compare_stata(key)
    else:
        print("\n未找到 Stata；已跳过 Stata 对照。可使用 --stata 指定路径。")

    print("\n=== 完成 ===")
    print(f"Python 输出: {OUT_PY}")
    if stata:
        print(f"Stata 输出:  {OUT_ST}")
    return 0 if (ok_py and ok_st) else 1


if __name__ == "__main__":
    raise SystemExit(main())
