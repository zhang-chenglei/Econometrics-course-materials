"""第4章案例：AI 使用与课程成绩——回归模型扩展：函数形式与二元结果。

本脚本承担两件事：

1. 生成第4、5章共用的一份教学数据（第5章案例沿用同一份数据）；
2. 完成第4章的分析：用对数、二次项和交互项表达不同的经济关系。

第5章的控制变量与诊断分析见 ``ch05/01_ai_score_diagnostics``。

教学用模拟数据，数据生成过程人为设定且已知：

    ai     = 3 + 3.2*ability + 0.25*age - 1.2*female + v,  v ~ N(0, 3.5)
    score  = 56 + 1.20*ai - 0.060*ai^2 + 0.55*(ai × female)
             + 4.5*ability + 0.6*age - 1.8*female
             + 0.20*parent_edu + 0.15*family_income + u,
             u 的方差随 ai 增大（第5章诊断）

其中 parent_edu 与 family_income 共享一个家庭背景因子，因此高度相关（第5章诊断）。

输出（本脚本同级的 output/）：
- ch04_model_comparison.csv        四个递进模型的比较表
- ch04_representative_marginal_effects.csv 代表性AI取值处的分组边际效应
- ch04_group_ame.csv                男女两组的平均边际效应
- ch04-fig2-quadratic.png          二次项示意图
- ch04-fig3-interaction.png        交互项示意图
- ch04-fig7-ai-score-curve.png     案例：AI 使用时间与预测成绩
- ch04-fig8-ai-gender.png          案例：男女两组的 AI 使用—成绩预测线
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260915
N = 800

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_ai_score_form_specification.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 只把图写进插图库，结果表始终留在脚本同级的 output/
TABLE_DIR = HERE / "output"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 第4、5章共用的一份数据
# ---------------------------------------------------------------------------
ability = rng.normal(0, 1, size=N)                                  # 认知能力（第5章的遗漏变量）
age = np.clip(np.round(rng.normal(20, 1.6, size=N)), 17, 26)        # 年龄
female = rng.integers(0, 2, size=N)                                 # 性别

# 家庭背景：父母受教育年限与家庭收入共享同一个因子，因此高度相关
family_bg = rng.normal(0, 1, size=N)
parent_edu = 12 + 2.2 * family_bg + rng.normal(0, 0.30, size=N)
family_income = 8 + 1.9 * family_bg + rng.normal(0, 0.30, size=N)

ai = 3 + 3.2 * ability + 0.25 * age - 1.2 * female + rng.normal(0, 3.5, size=N)
ai = np.clip(ai, 0, 30)                                             # 每周 AI 使用时间（小时）
ai2 = ai**2
ai_female = ai * female

# 误差方差随 AI 使用时间扩大（第5章用于演示异方差）
sigma = 2.0 + 0.35 * ai
score = (
    56
    + 1.20 * ai
    - 0.060 * ai2
    + 0.55 * ai_female
    + 4.5 * ability
    + 0.6 * age
    - 1.8 * female
    + 0.20 * parent_edu
    + 0.15 * family_income
    + rng.normal(0, sigma, size=N)
)

df = pd.DataFrame(
    {
        "score": score,
        "ai": ai,
        "ai2": ai2,
        "ai_female": ai_female,
        "age": age,
        "female": female,
        "parent_edu": parent_edu,
        "family_income": family_income,
        "ability": ability,
    }
)
df.to_csv(TABLE_DIR / "ch04_shared_data.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 四个递进模型：每一步增加一种表达能力
# ---------------------------------------------------------------------------
base = ["ai", "age", "female"]
df["ln_score"] = np.log(df["score"])
m_level = sm.OLS(df["score"], sm.add_constant(df[base])).fit()
m_log = sm.OLS(df["ln_score"], sm.add_constant(df[base])).fit()
m_quad = sm.OLS(df["ln_score"], sm.add_constant(df[base + ["ai2"]])).fit()
m_inter = sm.OLS(
    df["ln_score"], sm.add_constant(df[base + ["ai2", "ai_female"]])
).fit()

turning = -m_quad.params["ai"] / (2 * m_quad.params["ai2"])
cov = m_inter.cov_params()


def marginal_effect(ai_value: float, female_value: int) -> tuple[float, float]:
    """计算完整边际效应及 delta-method 标准误。"""
    gradient = pd.Series(0.0, index=m_inter.params.index)
    gradient.loc["ai"] = 1.0
    gradient.loc["ai2"] = 2.0 * ai_value
    gradient.loc["ai_female"] = float(female_value)
    effect = float(gradient @ m_inter.params)
    std_err = float(np.sqrt(gradient @ cov @ gradient))
    return effect, std_err

comparison = pd.DataFrame(
    {
        "model": ["成绩水平", "对数成绩", "+AI平方项", "+AI×性别"],
        "dependent_variable": ["score", "ln_score", "ln_score", "ln_score"],
        "ai_coef": [
            m_level.params["ai"],
            m_log.params["ai"],
            m_quad.params["ai"],
            m_inter.params["ai"],
        ],
        "r_squared": [m_level.rsquared, m_log.rsquared, m_quad.rsquared, m_inter.rsquared],
    }
)
comparison.to_csv(TABLE_DIR / "ch04_model_comparison.csv", index=False, encoding="utf-8-sig")

representative_rows = []
for ai_value in (5, 10, 15, 20):
    for female_value, group in ((0, "男性"), (1, "女性")):
        effect, std_err = marginal_effect(ai_value, female_value)
        representative_rows.append(
            {
                "ai_hours": ai_value,
                "female": female_value,
                "group": group,
                "marginal_effect": effect,
                "std_err": std_err,
                "ci_lower": effect - 1.96 * std_err,
                "ci_upper": effect + 1.96 * std_err,
            }
        )
representative_me = pd.DataFrame(representative_rows)
representative_me.to_csv(
    TABLE_DIR / "ch04_representative_marginal_effects.csv",
    index=False,
    encoding="utf-8-sig",
)

ame_rows = []
sample_mean_ai = float(df["ai"].mean())
for female_value, group in ((0, "男性"), (1, "女性")):
    # 按margins的标准口径：将female固定为指定组别，在全样本AI分布上取平均。
    effect, std_err = marginal_effect(sample_mean_ai, female_value)
    ame_rows.append(
        {
            "female": female_value,
            "group": group,
            "sample_mean_ai": sample_mean_ai,
            "average_marginal_effect": effect,
            "std_err": std_err,
            "ci_lower": effect - 1.96 * std_err,
            "ci_upper": effect + 1.96 * std_err,
        }
    )
group_ame = pd.DataFrame(ame_rows)
group_ame.to_csv(TABLE_DIR / "ch04_group_ame.csv", index=False, encoding="utf-8-sig")

print("模型比较（课程成绩；AI 使用时间）：")
print(comparison.round(5).to_string(index=False))
print(f"\n对数成绩模型中 AI 的极值点：{turning:.2f} 小时/周")
print("\n代表性AI取值处的完整边际效应（对数成绩）：")
print(representative_me.round(5).to_string(index=False))
print("\n分组平均边际效应（对数成绩）：")
print(group_ame.round(5).to_string(index=False))
print("\n样本中 AI 使用时间：均值 %.2f，范围 %.1f–%.1f" % (df.ai.mean(), df.ai.min(), df.ai.max()))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, GREY = "#B23A48", "#2A6F97", "#888888"

# 图4-2 二次项示意图：成绩随 AI 使用时间先升后降
grid = np.linspace(0, 30, 200)
quad_fit = sm.OLS(df["score"], sm.add_constant(df[["ai", "ai2"]])).fit()
pred = quad_fit.params["const"] + quad_fit.params["ai"] * grid + quad_fit.params["ai2"] * grid**2
turn_score = -quad_fit.params["ai"] / (2 * quad_fit.params["ai2"])
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.ai, df.score, s=12, alpha=0.20, color=GREY)
ax.plot(grid, pred, color=RED, linewidth=2.5)
ax.axvline(turn_score, color=BLUE, linestyle="--", label=f"转折点≈{turn_score:.1f} 小时/周")
ax.set(title="成绩随 AI 使用时间的变化", xlabel="每周 AI 使用时间（小时）", ylabel="课程成绩")
ax.legend(frameon=False)
fig.tight_layout()
fig2 = OUTPUT_DIR / "ch04-fig2-quadratic.png"
fig.savefig(fig2, dpi=220, bbox_inches="tight")

# 图4-3 交互项示意图：男女两组的 AI 使用—成绩线
fig, ax = plt.subplots(figsize=(8, 4.8))
for female_value, label, color in [(0, "男性", BLUE), (1, "女性", "#D17B88")]:
    sub = df[df.female == female_value]
    ax.scatter(sub.ai, sub.score, s=10, alpha=0.18, color=color)
    inter_fit = sm.OLS(
        df["score"], sm.add_constant(df[["ai", "female", "ai_female"]])
    ).fit()
    xs = np.linspace(0, 30, 100)
    ys = (
        inter_fit.params["const"]
        + inter_fit.params["ai"] * xs
        + inter_fit.params["female"] * female_value
        + inter_fit.params["ai_female"] * xs * female_value
    )
    ax.plot(xs, ys, color=color, linewidth=2.5, label=label)
ax.set(title="两组不同的斜率：一个交互项的直观表达",
       xlabel="每周 AI 使用时间（小时）", ylabel="课程成绩")
ax.legend(frameon=False)
fig.tight_layout()
fig3 = OUTPUT_DIR / "ch04-fig3-interaction.png"
fig.savefig(fig3, dpi=220, bbox_inches="tight")

# 图4-7 案例：对数成绩模型下的 AI 使用曲线（含二次项）
exp_grid = np.linspace(df.ai.min(), df.ai.max(), 200)
pred_quad = pd.DataFrame(
    {
        "const": 1.0,
        "ai": exp_grid,
        "age": df.age.mean(),
        "female": 0,
        "ai2": exp_grid**2,
    }
)[m_quad.model.exog_names]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.ai, df.ln_score, s=12, alpha=0.18, color=GREY)
ax.plot(exp_grid, m_quad.predict(pred_quad), color=RED, linewidth=2.5)
ax.axvline(turning, color=BLUE, linestyle="--", label=f"极值点≈{turning:.1f} 小时/周")
ax.set(title="AI 使用时间与预测对数成绩", xlabel="每周 AI 使用时间（小时）", ylabel="预测对数成绩")
ax.legend(frameon=False)
fig.tight_layout()
fig4 = OUTPUT_DIR / "ch04-fig7-ai-score-curve.png"
fig.savefig(fig4, dpi=220, bbox_inches="tight")

# 图4-8 案例：男女两组的 AI 使用—预测对数成绩线
fig, ax = plt.subplots(figsize=(8, 4.8))
for female_value, label, color in [(0, "男性", BLUE), (1, "女性", "#D17B88")]:
    pred = pd.DataFrame(
        {
            "const": 1.0,
            "ai": exp_grid,
            "age": df.age.mean(),
            "female": female_value,
            "ai2": exp_grid**2,
            "ai_female": exp_grid * female_value,
        }
    )[m_inter.model.exog_names]
    ax.plot(exp_grid, m_inter.predict(pred), label=label, color=color, linewidth=2.5)
ax.set(title="AI 使用与成绩：不同性别的两条预测曲线",
       xlabel="每周 AI 使用时间（小时）", ylabel="预测对数成绩")
ax.legend(frameon=False)
fig.tight_layout()
fig5 = OUTPUT_DIR / "ch04-fig8-ai-gender.png"
fig.savefig(fig5, dpi=220, bbox_inches="tight")

print(f"\n图形已保存：{fig2}\n图形已保存：{fig3}\n图形已保存：{fig4}\n图形已保存：{fig5}")
