#!/usr/bin/env python3
"""在临时副本中运行论文七张表、两幅图；保留原始数据与代码。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "original"
STEPS = [f"maketable{i}" for i in range(1, 8)] + ["makefigure1", "makefigure2"]
EXPECTED = ["table1.out", "Table2cluster.txt", "table3.txt", "table3_rich.txt",
            "Table4.out", "table5spatial.out", "071004_20yrs_firstyear_table6.txt",
            "table7.txt", "-temp-presentation.eps", "-precip-presentation.eps", "Fig2.eps"]


def hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def patch_exports(work: Path) -> list[dict]:
    """当前 outreg 已重写接口；仅替换导出命令，不改估计、样本与检验。"""
    changes = []
    for name in ["maketable3.do", "maketable6.do"]:
        p = work / name
        old = p.read_text()
        new, count = re.subn(r"(?m)^(\s*)outreg\s", r"\1outreg2 ", old)
        new = new.replace("3aster coefastr se", "se bdec(3) alpha(0.01,0.05,0.10)")
        # 逐行核对：所有差异必须只发生在 outreg 导出行。
        assert len(old.splitlines()) == len(new.splitlines())
        assert all(a == b or a.lstrip().startswith("outreg ")
                   for a, b in zip(old.splitlines(), new.splitlines()))
        p.write_text(new)
        changes.append({"file": name, "export_lines": count,
                        "change": "outreg 旧接口改为 outreg2；显式设置三位小数、标准误与1/5/10%星号"})
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stata", default=os.environ.get("STATA_BIN") or shutil.which("stata-mp")
                        or "")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stata = Path(args.stata).expanduser().resolve()
    if not stata.is_file():
        parser.error("找不到 Stata，请用 --stata 指定可执行文件。")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    output = (args.output or ROOT / "output" / "original" / stamp).resolve()
    if output.exists():
        parser.error("输出目录已存在；请选择新目录，避免覆盖已有结果。")
    before = hashes(SOURCE)
    work = Path(tempfile.mkdtemp(prefix="lecture6-original-"))
    shutil.copytree(SOURCE, work, dirs_exist_ok=True)
    changes = patch_exports(work)
    lines = ["clear all", "set more off", "set linesize 255", 'display "L6_PREFLIGHT"']
    for command in ["mmerge", "cgmreg", "outreg", "outreg2"]:
        lines += [f"capture noisily which {command}", "local rc = _rc",
                  'if `rc\' != 0 exit `rc\'']
    for step in STEPS:
        lines += [f'capture noisily do "{step}.do"', "local rc = _rc",
                  'if `rc\' != 0 exit `rc\'', f'display "L6_STEP_OK:{step}"']
    lines.append('display "L6_ALL_OK"')
    (work / "verify.do").write_text("\n".join(lines) + "\n")
    print(f"运行副本：{work}\n结果目录：{output}", flush=True)
    process = subprocess.run([str(stata), "-b", "do", "verify.do"], cwd=work, capture_output=True)
    logpath = work / "verify.log"
    log = logpath.read_text(errors="replace") if logpath.exists() else ""
    # 去掉 Stata 的许可证抬头；仅保存命令开始后的运行记录。
    start = log.find(". do verify.do")
    cleanlog = log[start:] if start >= 0 else log[log.find(". clear all"):]
    completed = re.findall(r"(?m)^L6_STEP_OK:(\w+)\s*$", cleanlog)
    missing = [name for name in EXPECTED if not (work / name).is_file() or (work / name).stat().st_size == 0]
    unchanged = before == hashes(SOURCE)
    success = (process.returncode == 0 and completed == STEPS and
               bool(re.search(r"(?m)^L6_ALL_OK\s*$", cleanlog)) and not missing and unchanged)
    output.mkdir(parents=True)
    (output / "运行日志.log").write_text(cleanlog)
    for p in work.iterdir():
        if p.name not in before and p.suffix.lower() in {".txt", ".out", ".xml", ".eps", ".gph"}:
            shutil.copy2(p, output / p.name)
    code = output / "运行代码"
    code.mkdir()
    for name in ["master.do", "verify.do"] + [s + ".do" for s in STEPS]:
        shutil.copy2(work / name, code / name)
    metadata = {"time": stamp, "success": success, "stata": str(stata), "work_copy": str(work),
                "completed_steps": completed, "missing_outputs": missing,
                "original_files_unchanged": unchanged, "compatibility_changes": changes,
                "source_sha256": before, "process_returncode": process.returncode}
    (output / "运行核验.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"完成 {len(completed)}/9 个步骤；原始文件未变：{unchanged}；通过：{success}", flush=True)
    if not success:
        print("请查看运行日志末尾与运行核验.json；失败的运行副本已保留。")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
