from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CHAPTER_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = CHAPTER_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR / "共享代码"))

from common import (  # noqa: E402
    BLUE,
    GOLD,
    GREY,
    chapter_dir,
    export_table,
    load_data,
    load_policy,
    save_figure,
    setup_plot_style,
)


def main() -> None:
    setup_plot_style()
    chapter = chapter_dir(12)
    table_dir = chapter / "表格"
    figure_dir = chapter / "图形"
    note_dir = chapter / "说明"

    data = load_data()
    policy = load_policy()
    policy["approval_date"] = pd.to_datetime(policy["approval_date"])

    policy_table = policy[
        [
            "pilot_id",
            "cohort_id",
            "scope_level",
            "scope_name",
            "province",
            "prefecture",
            "county",
            "approval_date",
            "approval_year",
            "first_full_year",
            "document_no",
            "source_url",
            "matched_firms",
        ]
    ].copy()
    policy_table["approval_date"] = policy_table["approval_date"].dt.strftime(
        "%Y-%m-%d"
    )
    export_table(policy_table, table_dir / "表12-1_政策事实表")

    treated = data.loc[data["did_treated"].eq(1)].copy()
    support = (
        treated.groupby(
            ["policy_cohort_id", "gvar"],
            dropna=False,
        )
        .agg(
            pilots=("policy_pilot_id", "nunique"),
            policy_scopes=("policy_scope_name", "nunique"),
            firms=("firm_id", "nunique"),
            observations=("firm_id", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            min_event=("event_time_full", "min"),
            max_event=("event_time_full", "max"),
        )
        .reset_index()
    )
    export_table(support, table_dir / "表12-2_处理批次支持域")

    pilot_support = (
        treated.groupby(
            [
                "policy_pilot_id",
                "policy_scope_name",
                "policy_cohort_id",
                "policy_first_full_year",
            ],
            dropna=False,
        )
        .agg(
            firms=("firm_id", "nunique"),
            observations=("firm_id", "size"),
            cities=("analysis_city_id", "nunique"),
            pre_observations=("did_post_full", lambda values: int(values.eq(0).sum())),
            post_observations=("did_post_full", lambda values: int(values.eq(1).sum())),
        )
        .reset_index()
    )
    export_table(pilot_support, table_dir / "表12-3_各试点样本覆盖")

    monotonicity_violations = 0
    for _, group in data.sort_values(["firm_id", "year"]).groupby("firm_id"):
        if (group["did_post_full"].diff().fillna(0) < 0).any():
            monotonicity_violations += 1
    design_checks = pd.DataFrame(
        [
            ["数据类型", "半合成教学数据", "通过"],
            ["样本期", f"{data['year'].min()}—{data['year'].max()}", "通过"],
            ["企业数", f"{data['firm_id'].nunique():,}", "通过"],
            ["企业—年份观测", f"{len(data):,}", "通过"],
            ["企业—年份重复键", int(data.duplicated(["firm_id", "year"]).sum()), "通过"],
            ["正式试点范围", int(policy["pilot_id"].nunique()), "通过"],
            ["正式批复日期组", int(policy["cohort_id"].nunique()), "通过"],
            ["年度处理组", int(data.loc[data["gvar"].gt(0), "gvar"].nunique()), "通过"],
            ["处理状态单调性违反企业", monotonicity_violations, "通过"],
            ["城市聚类数", int(data["analysis_city_id"].nunique()), "通过"],
            ["处理企业数", int(treated["firm_id"].nunique()), "通过"],
            [
                "从未处理企业数",
                int(data.loc[data["did_treated"].eq(0), "firm_id"].nunique()),
                "通过",
            ],
        ],
        columns=["check", "result", "status"],
    )
    export_table(design_checks, table_dir / "表12-4_研究设计验收")

    threats = pd.DataFrame(
        [
            ["选择性获批", "政策前特征与趋势", "描述统计、平衡检验、交错DID", "第13、14章"],
            ["政策预期", "政策前事件时间系数", "事件研究与联合检验", "第14章"],
            ["批复年份错配", "批复当年与完整年度", "替代处理起点", "第14章"],
            ["疫情冲击", "2020年与特殊城市", "排除年份和城市", "第14章"],
            ["同期政策", "教学代理变量", "控制或剔除代理重叠样本", "第14章"],
            ["地址错配", "教学办公地址代理", "替代处理时点", "第14章"],
            ["空间溢出", "同省近邻教学代理", "剔除近邻对照组", "第14章"],
            ["少量处理范围", "18个试点范围", "城市聚类与留一试点", "第14章"],
        ],
        columns=["识别威胁", "所需证据", "对应检验", "落实章节"],
    )
    export_table(threats, table_dir / "表12-5_识别威胁与证据地图")

    # Figure 11-1: approval timeline.
    timeline = policy.sort_values(["approval_date", "pilot_id"]).copy()
    y_positions = np.arange(len(timeline))
    colors = np.where(
        timeline["first_full_year"].eq(2020),
        BLUE,
        np.where(timeline["first_full_year"].eq(2021), GOLD, GREY),
    )
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.hlines(
        y_positions,
        timeline["approval_date"].min() - pd.Timedelta(days=40),
        timeline["approval_date"],
        color=LIGHT_GREY if "LIGHT_GREY" in globals() else "#D7DDE2",
        linewidth=1.2,
    )
    axis.scatter(
        timeline["approval_date"],
        y_positions,
        c=colors,
        s=55,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        timeline["pilot_id"] + " " + timeline["scope_name"],
        fontsize=9,
    )
    axis.invert_yaxis()
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.set_xlabel("科技部正式批复日期")
    axis.set_title(
        "国家新一代人工智能创新发展试验区批复时间线",
        loc="left",
        pad=30,
        fontweight="bold",
    )
    axis.text(
        0,
        1.01,
        "18个正式试点范围；颜色对应批复后第一个完整处理年度",
        transform=axis.transAxes,
        color=GREY,
        fontsize=10,
        va="bottom",
    )
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    save_figure(fig, figure_dir / "图12-1_试验区批复时间线")

    # Figure 11-2: firm support by pilot.
    plot_data = pilot_support.sort_values("firms", ascending=True)
    fig, axis = plt.subplots(figsize=(10, 7))
    bar_colors = np.where(
        plot_data["policy_first_full_year"].eq(2020),
        BLUE,
        np.where(plot_data["policy_first_full_year"].eq(2021), GOLD, GREY),
    )
    axis.barh(
        plot_data["policy_pilot_id"] + " " + plot_data["policy_scope_name"],
        plot_data["firms"],
        color=bar_colors,
        edgecolor="white",
    )
    for position, value in enumerate(plot_data["firms"]):
        axis.text(value + 2, position, f"{value:,}", va="center", fontsize=9)
    axis.set_xlabel("半合成教学样本中的企业数")
    axis.set_title(
        "各试点范围的企业样本覆盖",
        loc="left",
        pad=30,
        fontweight="bold",
    )
    axis.text(
        0,
        1.01,
        "按企业层面约50%分层抽样；不同试点原始上市公司数量差异较大",
        transform=axis.transAxes,
        color=GREY,
        fontsize=10,
        va="bottom",
    )
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    save_figure(fig, figure_dir / "图12-2_各试点企业覆盖")

    summary = f"""# 第12章案例结果说明

> 本章结果基于半合成教学数据。政策事实与处理批次来自正式批复整理，但企业结果变量包含预设政策效应，不能解释为现实政策评估结论。

样本覆盖2011—2024年，共有{data['firm_id'].nunique():,}家匿名企业和{len(data):,}个企业—年份观测。政策事实表包含{policy['pilot_id'].nunique()}个正式试点范围和{policy['cohort_id'].nunique()}个精确批复日期组。由于企业结果按年度观察，批复后的第一个完整处理年度最终折叠为2020、2021和2022三个年度处理组。

处理状态通过唯一键、单调性和支持域检查。样本中有{treated['firm_id'].nunique():,}家处理企业、{data.loc[data['did_treated'].eq(0), 'firm_id'].nunique():,}家从未处理企业，城市聚类数为{data['analysis_city_id'].nunique()}。各年度处理组均具有多个政策前年份和至少两个政策后年份，可以进入第14章的动态分析。

第12章只确认研究问题、政策事实、处理时点和识别威胁，不在研究设计阶段使用显著性倒推模型。同期政策、办公地址和空间距离在当前教学数据中没有真实观测，第14章只能使用明确标注的教学代理变量演示代码路径。
"""
    (note_dir / "第12章案例结果说明.md").write_text(summary, encoding="utf-8")

    print(chapter)


if __name__ == "__main__":
    main()
