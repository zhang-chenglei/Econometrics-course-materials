#!/usr/bin/env python3
"""Lesson 4 case: AI use, student scores, and causal identification.

Download ``AI_score_case.csv`` from the course page and place it beside this
script, or pass its location with ``--data``. The script describes the sample,
draws the main scatter plot, and compares the naive and adjusted regressions.
The regressions are descriptive; they are not a causal identification strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to AI_score_case.csv (default: beside this script or under data/ai-score-case from the current directory)",
    )
    return parser.parse_args()


def find_data(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser())
    candidates.extend(
        [
            Path(__file__).resolve().parent / "AI_score_case.csv",
            Path.cwd() / "AI_score_case.csv",
            Path.cwd() / "data" / "ai-score-case" / "AI_score_case.csv",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "找不到 AI_score_case.csv。请从课程页面下载后放在脚本旁边，"
        "或使用 --data 指定路径。\n已检查：\n" + tried
    )


def main() -> None:
    args = parse_args()
    data_path = find_data(args.data)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    data = pd.read_csv(data_path)
    variables = [
        "final_score",
        "ai_hours",
        "prior_score",
        "motivation",
        "digital_skill",
        "study_hours",
        "assignment_quality",
    ]

    print(f"读取数据：{data_path}")
    print("\n描述统计")
    print(data[variables].describe().T.round(3))
    print("\nAI 使用与期末成绩的相关系数")
    print(data[["ai_hours", "final_score"]].corr().round(3))

    naive = smf.ols("final_score ~ ai_hours", data=data).fit(cov_type="HC1")
    adjusted = smf.ols(
        "final_score ~ ai_hours + prior_score + motivation + digital_skill "
        "+ study_hours + C(gender) + C(class_id)",
        data=data,
    ).fit(cov_type="HC1")

    comparison = pd.DataFrame(
        {
            "model": ["naive", "adjusted"],
            "ai_coefficient": [naive.params["ai_hours"], adjusted.params["ai_hours"]],
            "robust_se": [naive.bse["ai_hours"], adjusted.bse["ai_hours"]],
            "p_value": [naive.pvalues["ai_hours"], adjusted.pvalues["ai_hours"]],
            "r_squared": [naive.rsquared, adjusted.rsquared],
            "n": [int(naive.nobs), int(adjusted.nobs)],
        }
    )
    print("\n两列回归比较（仍然是条件相关，不是因果识别）")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(output_dir / "regression_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(data["ai_hours"], data["final_score"], alpha=0.28, s=22)
    ordered = data.sort_values("ai_hours")
    axis.plot(
        ordered["ai_hours"],
        naive.predict(ordered),
        color="#941f3a",
        linewidth=2,
        label="simple fitted line",
    )
    axis.set_xlabel("Weekly AI-use hours")
    axis.set_ylabel("Final score")
    axis.set_title("AI use and final score: correlation, not yet causation")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "ai_score_scatter.png", dpi=180)
    plt.close(figure)

    print(f"\n结果已保存到：{output_dir}")
    print("下一步不是继续堆控制变量，而是写清反事实并重新设计研究。")


if __name__ == "__main__":
    main()
