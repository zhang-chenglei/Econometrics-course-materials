"""案例：一份工资回归的完整诊断。

生成一张图和四份结果表，全部保存到本脚本同级的 output/ 文件夹。
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

SEED = 20260912
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_model_diagnostics.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

n = 800
ability = rng.normal(size=n)
educ = np.clip(14 + 0.8 * ability + rng.normal(0, 1.4, size=n), 9, 22)
exper = np.clip(rng.normal(14, 6, size=n), 0, 35)
female = rng.integers(0, 2, size=n)
city_factor = rng.normal(size=n)
city_income = city_factor + rng.normal(0, 0.15, size=n)
city_size = city_factor + rng.normal(0, 0.15, size=n)

# 方差随经验增加，制造可见的异方差。
sigma = 0.18 + 0.018 * exper
u = rng.normal(0, sigma, size=n)
ln_wage = (
    2.0
    + 0.08 * educ
    + 0.035 * exper
    - 0.10 * female
    + 0.10 * ability
    + 0.05 * city_income
    + 0.04 * city_size
    + u
)
df = pd.DataFrame(
    {
        "ln_wage": ln_wage,
        "educ": educ,
        "exper": exper,
        "female": female,
        "city_income": city_income,
        "city_size": city_size,
        "ability": ability,
    }
)

naive_vars = ["educ", "exper", "female", "city_income", "city_size"]
full_vars = naive_vars + ["ability"]
naive = sm.OLS(df["ln_wage"], sm.add_constant(df[naive_vars])).fit()
full = sm.OLS(df["ln_wage"], sm.add_constant(df[full_vars])).fit()
robust = naive.get_robustcov_results(cov_type="HC1")

# 任务1：遗漏变量比较
ovb_table = pd.DataFrame(
    {
        "model": ["遗漏ability", "控制ability"],
        "educ_coef": [naive.params["educ"], full.params["educ"]],
        "educ_se": [naive.bse["educ"], full.bse["educ"]],
    }
)

# 任务2：VIF
X_vif = sm.add_constant(df[full_vars])
vif_table = pd.DataFrame(
    {
        "variable": X_vif.columns,
        "vif": [variance_inflation_factor(X_vif.to_numpy(), i) for i in range(X_vif.shape[1])],
    }
)
vif_table.to_csv(OUTPUT_DIR / "ch05_vif.csv", index=False, encoding="utf-8-sig")

# 任务3：异方差与稳健标准误
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
final_table.to_csv(OUTPUT_DIR / "ch05_final_regression.csv", encoding="utf-8-sig")

diagnostic = pd.DataFrame(
    [
        ["遗漏ability", "教育系数可能有偏", "比较参照模型", "现实中ability通常不可观测"],
        ["城市变量高度相关", "系数精度下降", "结合研究问题解释VIF", "不能机械按阈值删变量"],
        ["异方差", "常规标准误可能失效", "报告HC1稳健标准误", "不能修复遗漏变量偏误"],
    ],
    columns=["问题", "主要影响", "本案例处理", "处理后的限制"],
)
diagnostic.to_csv(OUTPUT_DIR / "ch05_diagnostic_checklist.csv", index=False, encoding="utf-8-sig")

print("遗漏变量比较：")
print(ovb_table.round(5).to_string(index=False))
print("\nVIF：")
print(vif_table.round(3).to_string(index=False))
print(f"\nBreusch-Pagan p值：{bp_lm_p:.6g}")
print(f"White检验 p值：{white_lm_p:.6g}")
print("\n常规标准误与稳健标准误：")
print(final_table.round(5))
print("\n诊断清单：")
print(diagnostic.to_string(index=False))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(naive.fittedvalues, naive.resid, s=15, alpha=0.35, color="#2A6F97")
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set(title="工资模型残差图：误差波动并不恒定", xlabel="拟合值", ylabel="残差")
fig.tight_layout()
output = OUTPUT_DIR / "ch05-fig3-residual-heteroskedasticity.png"
fig.savefig(output, dpi=220, bbox_inches="tight")
print(f"\n图形已保存：{output}")
