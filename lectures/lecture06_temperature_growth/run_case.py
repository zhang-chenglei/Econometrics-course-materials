#!/usr/bin/env python3
"""第六讲案例入口；保留整个目录，所有新结果写入同级 output/。"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def locate_stata(explicit):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise SystemExit("--stata 指定的文件不存在。")
        return p
    for name in ["stata-mp", "stata-se", "stata", "StataMP-64.exe", "StataSE-64.exe"]:
        if p := shutil.which(name):
            return Path(p)
    if sys.platform == "darwin":
        candidates = sorted(Path("/Applications").glob("Stata*/Stata*.app/Contents/MacOS/stata-mp"))
        candidates += sorted(Path("/Applications").glob("Stata*/Stata*.app/Contents/MacOS/stata-se"))
        candidates += sorted(Path("/Applications").glob("Stata*/Stata*.app/Contents/MacOS/stata"))
        if candidates:
            return candidates[-1]
    raise SystemExit("未找到 Stata；请用 --stata 指定可执行文件，或设置 STATA_BIN。")


def teaching(stata, install):
    (ROOT / "output/data").mkdir(parents=True, exist_ok=True)
    code = "set more off\nset linesize 255\n"
    if install:
        code += 'do "scripts/install_dependencies.do"\n'
    code += 'do "scripts/build_panel.do"\ndisplay "L6_TEACHING_OK"\n'
    # Stata 批处理退出码不足以证明运行成功，另外检查明确的完成标记。
    (ROOT / "output/run_teaching.do").write_text(code)
    proc = subprocess.run([str(stata), "-b", "do", "output/run_teaching.do"], cwd=ROOT, capture_output=True)
    raw = ROOT / "run_teaching.log"
    text = raw.read_text(errors="replace") if raw.exists() else ""
    start = text.find(". do output/run_teaching.do")
    clean = text[start:] if start >= 0 else text[text.find(". set more off"):]
    (ROOT / "output/teaching.log").write_text(clean)
    if raw.exists():
        # 移入 output，而不是在包根留下含许可证抬头的日志。
        raw.replace(ROOT / "output/stata_batch.log")
    if proc.returncode or not re.search(r"(?m)^L6_TEACHING_OK\s*$", clean):
        raise SystemExit("教学流程未完成；请查看 output/teaching.log 的末尾。")
    for name in ["build_numbers.py", "generate_figures.py"]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)
    with (ROOT / "output/data/regression_results.csv").open() as f:
        actual = list(csv.DictReader(f))
    with (ROOT / "reference/regression_results.csv").open() as f:
        expected = list(csv.DictReader(f))
    key = lambda row: (row["spec"], row["term"])
    expected = {key(row): row for row in expected}
    differences = []
    for row in actual:
        ref = expected.get(key(row))
        if ref is None or any(abs(float(row[x])-float(ref[x])) > 1e-6 for x in ["b", "se", "p", "n"]):
            differences.append(key(row))
    matches = len(actual) == len(expected) and not differences
    report = {"time": datetime.now().astimezone().isoformat(), "completed": True,
              "reference_matches": matches, "regression_rows": len(actual), "different_terms": differences,
              "non_psd_adjustment_messages": clean.count("non-positive semi-definite")}
    (ROOT / "output/teaching_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"教学流程完成；{len(actual)} 条记录与参考值匹配：{matches}", flush=True)
    if not matches:
        raise SystemExit("运行已完成，但环境/结果与参考版本存在差异；请核查，不要直接修改星号。")


def install_dependencies(stata):
    (ROOT / "output").mkdir(exist_ok=True)
    proc = subprocess.run([str(stata), "-b", "do", "scripts/install_dependencies.do"], cwd=ROOT, capture_output=True)
    raw = ROOT / "install_dependencies.log"
    text = raw.read_text(errors="replace") if raw.exists() else ""
    if raw.exists():
        raw.replace(ROOT / "output/install_dependencies.log")
    if proc.returncode or not re.search(r"(?m)^L6_DEPENDENCIES_OK\s*$", text):
        raise SystemExit("依赖核验未完成；请查看 output/install_dependencies.log 的末尾。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["teaching", "original", "all"], default="teaching")
    parser.add_argument("--stata", default=os.environ.get("STATA_BIN"))
    parser.add_argument("--install", action="store_true", help="先安装/核验 Stata 依赖")
    args = parser.parse_args()
    stata = locate_stata(args.stata)
    if args.mode in ["teaching", "all"]:
        teaching(stata, args.install)
    elif args.install:
        install_dependencies(stata)
    if args.mode in ["original", "all"]:
        subprocess.run([sys.executable, str(ROOT / "scripts/run_original_replication.py"),
                        "--stata", str(stata)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
