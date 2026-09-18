"""第5章案例：AI 使用与课程成绩——模型设定与诊断。

本脚本沿用第4章案例的同一份教学数据（同一种子、同一份生成过程），完成两件事：

1. 控制变量到底在控制什么：遗漏 ability 与加入它的对比，并用 FWL 三步验证
   “净 variation 解释净 variation”；
2. 模型诊断：多重共线性（parent_edu 与 family_income）、异方差（误差方差随 ai 扩大）。

第4章的模型形式分析见 ``ch04/01_ai_score_form_specification``。

输出（本脚本同级的 output/）：
- ch05_ovb_comparison.csv          遗漏/控制 ability 时的 AI 系数
- ch05_fwl_check.csv               FWL 三步回归的系数对照
- ch05_vif.csv                     各解释变量的 VIF
- ch05_final_regression.csv        常规标准误与 HC1 稳健标准误
- ch05_diagnostic_checklist.csv    四列诊断清单
- ch05-fig3-residual-heteroskedasticity.png 残差图
- ch05-fig4-fwl-net-variation.png  净化后的 AI 与净化后的成绩
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 20260915
N = 800

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_ai_score_diagnostics.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 只把图写进插图库，结果表始终留在脚本同级的 output/
TABLE_DIR = HERE / "output"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 与第4章共用的一份数据（生成块逐字一致）
# ---------------------------------------------------------------------------
ability = rng.normal(0, 1, size=N)
age = np.clip(np.round(rng.normal(20, 1.6, size=N)), 17, 26)
female = rng.integers(0, 2, size=N)

family_bg = rng.normal(0, 1, size=N)
parent_edu = 12 + 2.2 * family_bg + rng.normal(0, 0.30, size=N)
family_income = 8 + 1.9 * family_bg + rng.normal(0, 0.30, size=N)

ai = 3 + 3.2 * ability + 0.25 * age - 1.2 * female + rng.normal(0, 3.5, size=N)
ai = np.clip(ai, 0, 30)
ai2 = ai**2
ai_female = ai * female

sigma = 2.0 + 0.35 * ai
score = (
    56
    + 2.0 * ai
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

# 本段为了讨论“控制”而使用对 ai 线性的设定；函数形式问题已在第4章处理。
naive_vars = ["ai", "age", "female", "parent_edu", "family_income"]
full_vars = naive_vars + ["ability"]
naive = sm.OLS(df["score"], sm.add_constant(df[naive_vars])).fit()
full = sm.OLS(df["score"], sm.add_constant(df[full_vars])).fit()
robust = naive.get_robustcov_results(cov_type="HC1")

# ===========================================================================
# 第一部分：控制变量到底在控制什么
# ===========================================================================
ovb_table = pd.DataFrame(
    {
        "model": ["遗漏ability", "控制ability"],
        "ai_coef": [naive.params["ai"], full.params["ai"]],
        "ai_se": [naive.bse["ai"], full.bse["ai"]],
    }
)
ovb_table.to_csv(TABLE_DIR / "ch05_ovb_comparison.csv", index=False, encoding="utf-8-sig")

# FWL：控制变量的集合就是多元回归中除 ai 之外的全部解释变量
z_vars = ["age", "female", "parent_edu", "family_income", "ability"]
ai_resid = sm.OLS(df["ai"], sm.add_constant(df[z_vars])).fit().resid
score_resid = sm.OLS(df["score"], sm.add_constant(df[z_vars])).fit().resid
# 两个残差的均值都是 0，故不再加常数项
fwl = sm.OLS(score_resid, ai_resid).fit()
fwl_table = pd.DataFrame(
    {
        "path": ["多元回归：ai 的系数", "FWL 第 3 步：净成绩对净 AI 回归"],
        "ai_coef": [full.params["ai"], fwl.params.iloc[0]],
    }
)
fwl_table.to_csv(TABLE_DIR / "ch05_fwl_check.csv", index=False, encoding="utf-8-sig")

# ===========================================================================
# 第二部分：模型诊断
# ===========================================================================
X_vif = sm.add_constant(df[full_vars])
# 常数项的 VIF 没有解释意义，只报告解释变量
vif_table = pd.DataFrame(
    {
        "variable": full_vars,
        "vif": [
            variance_inflation_factor(X_vif.to_numpy(), X_vif.columns.get_loc(name))
            for name in full_vars
        ],
    }
)
vif_table.to_csv(TABLE_DIR / "ch05_vif.csv", index=False, encoding="utf-8-sig")

bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(naive.resid, naive.model.exog)
white_lm, white_lm_p, white_f, white_f_p = het_white(naive.resid, naive.model.exog)
robust_bse = pd.Series(robust.bse, index=naive.params.index)
final_table = pd.DataFrame(
    {
        "coef": naive.params,
        "conventional_se": naive.bse,
        "robust_se_HC1": robust_bse,
    }
)
final_table.to_csv(TABLE_DIR / "ch05_final_regression.csv", encoding="utf-8-sig")

# 函数形式的诊断工具：RESET 检验（在朴素模型上追加拟合值的幂次）
reset = sm.OLS(
    df["score"],
    sm.add_constant(pd.concat([df[naive_vars], (naive.fittedvalues**2).rename("fit2"),
                               (naive.fittedvalues**3).rename("fit3")], axis=1)),
).fit()
reset_f = reset.f_test("fit2 = 0, fit3 = 0")

diagnostic = pd.DataFrame(
    [
        ["函数形式错误", "E(Y|X) 的设定", "残差图 / RESET / 经济理论", "重新设定函数形式（对数、二次项、交互项）"],
        ["异方差", "标准误、t、p 与置信区间", "残差图 / BP / White", "报告 HC1 稳健标准误"],
        ["多重共线性", "单个系数的精度与稳定性", "相关系数 / VIF", "不机械删变量，回到研究问题找有效 variation"],
        ["遗漏 ability", "AI 系数本身（偏误）", "不能靠普通诊断证明", "换识别策略，留到因果推断再讲"],
    ],
    columns=["问题", "主要伤害什么", "怎么发现", "怎么处理"],
)
diagnostic.to_csv(TABLE_DIR / "ch05_diagnostic_checklist.csv", index=False, encoding="utf-8-sig")

print("第一部分 控制变量：")
print(ovb_table.round(5).to_string(index=False))
print("\nFWL 检验：")
print(fwl_table.round(8).to_string(index=False))
print(f"两步系数之差：{abs(full.params['ai'] - fwl.params.iloc[0]):.3e}")
print("\n变量相关系数（家庭背景两项）：")
print(f"  parent_edu 与 family_income = {df['parent_edu'].corr(df['family_income']):.4f}")
print("\n第二部分 诊断 —— VIF：")
print(vif_table.round(3).to_string(index=False))
print(f"\nBreusch-Pagan p值：{bp_lm_p:.6g}")
print(f"White检验 p值：{white_lm_p:.6g}")
print(f"RESET 检验 F={float(reset_f.fvalue):.3f}, p={float(reset_f.pvalue):.6g}")
print("\n常规标准误与稳健标准误：")
print(final_table.round(5))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, GREY = "#B23A48", "#2A6F97", "#888888"

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(naive.fittedvalues, naive.resid, s=15, alpha=0.35, color=BLUE)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set(title="成绩模型残差图：误差波动并不恒定", xlabel="拟合值", ylabel="残差")
fig.tight_layout()
output = OUTPUT_DIR / "ch05-fig3-residual-heteroskedasticity.png"
fig.savefig(output, dpi=220, bbox_inches="tight")

# 净化后的 AI 对净化后的成绩：斜率就是多元回归中的 AI 系数
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(ai_resid, score_resid, s=15, alpha=0.35, color=GREY)
grid = np.linspace(ai_resid.min(), ai_resid.max(), 2)
ax.plot(grid, fwl.params.iloc[0] * grid, color=RED, linewidth=2.5,
        label=f"斜率 = {fwl.params.iloc[0]:.4f}")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(title="净化后的 AI 使用时间对净化后的成绩",
       xlabel="净 AI 使用时间（剔除其他变量能解释的部分）",
       ylabel="净成绩 variation")
ax.legend(frameon=False)
fig.tight_layout()
output_fwl = OUTPUT_DIR / "ch05-fig4-fwl-net-variation.png"
fig.savefig(output_fwl, dpi=220, bbox_inches="tight")

print(f"\n图形已保存：{output}\n图形已保存：{output_fwl}")
