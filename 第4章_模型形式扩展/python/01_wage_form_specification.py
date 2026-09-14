"""案例：从直线到曲线——工资模型的形式扩展。

生成两张图和两份结果表，全部保存到本脚本同级的 output/ 文件夹。
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260911
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_wage_form_specification.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

n = 800
female = rng.integers(0, 2, size=n)
educ = np.clip(rng.normal(14, 2, size=n), 9, 22)
exper = np.clip(rng.normal(15, 7, size=n), 0, 38)
exper2 = exper**2
u = rng.normal(0, 0.28, size=n)

ln_wage = (
    5.0
    + 0.075 * educ
    + 0.050 * exper
    - 0.0009 * exper2
    - 0.12 * female
    + 0.018 * educ * female
    + u
)
wage = np.exp(ln_wage)
df = pd.DataFrame(
    {
        "wage": wage,
        "ln_wage": ln_wage,
        "educ": educ,
        "exper": exper,
        "exper2": exper2,
        "female": female,
        "educ_female": educ * female,
    }
)

base = ["educ", "exper", "female"]
m_level = sm.OLS(df["wage"], sm.add_constant(df[base])).fit()
m_log = sm.OLS(df["ln_wage"], sm.add_constant(df[base])).fit()
m_quad = sm.OLS(df["ln_wage"], sm.add_constant(df[base + ["exper2"]])).fit()
m_inter = sm.OLS(
    df["ln_wage"], sm.add_constant(df[base + ["exper2", "educ_female"]])
).fit()

turning = -m_quad.params["exper"] / (2 * m_quad.params["exper2"])
cov = m_inter.cov_params()
male_slope = m_inter.params["educ"]
female_slope = m_inter.params["educ"] + m_inter.params["educ_female"]
male_se = np.sqrt(cov.loc["educ", "educ"])
female_se = np.sqrt(
    cov.loc["educ", "educ"]
    + cov.loc["educ_female", "educ_female"]
    + 2 * cov.loc["educ", "educ_female"]
)

comparison = pd.DataFrame(
    {
        "model": ["工资水平", "对数工资", "+经验平方项", "+教育×女性"],
        "dependent_variable": ["wage", "ln_wage", "ln_wage", "ln_wage"],
        "educ_coef": [m_level.params["educ"], m_log.params["educ"], m_quad.params["educ"], m_inter.params["educ"]],
        "r_squared": [m_level.rsquared, m_log.rsquared, m_quad.rsquared, m_inter.rsquared],
    }
)
comparison.to_csv(OUTPUT_DIR / "ch04_model_comparison.csv", index=False, encoding="utf-8-sig")
group_slopes = pd.DataFrame(
    {
        "group": ["男性", "女性"],
        "educ_slope": [male_slope, female_slope],
        "std_err": [male_se, female_se],
    }
)
group_slopes["ci_lower"] = group_slopes.educ_slope - 1.96 * group_slopes.std_err
group_slopes["ci_upper"] = group_slopes.educ_slope + 1.96 * group_slopes.std_err
group_slopes.to_csv(OUTPUT_DIR / "ch04_group_slopes.csv", index=False, encoding="utf-8-sig")

print("模型比较：")
print(comparison.round(5).to_string(index=False))
print(f"\n经验曲线极值点：{turning:.2f}年")
print("\n男女教育斜率：")
print(group_slopes.round(5).to_string(index=False))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 经验曲线：其他变量固定在有意义的值
exp_grid = np.linspace(df.exper.min(), df.exper.max(), 200)
pred_quad = pd.DataFrame(
    {
        "const": 1.0,
        "educ": df.educ.mean(),
        "exper": exp_grid,
        "female": 0,
        "exper2": exp_grid**2,
    }
)[m_quad.model.exog_names]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.exper, df.ln_wage, s=12, alpha=0.18, color="#888888")
ax.plot(exp_grid, m_quad.predict(pred_quad), color="#B23A48", linewidth=2.5)
ax.axvline(turning, color="#2A6F97", linestyle="--", label=f"极值点≈{turning:.1f}年")
ax.set(title="工作经验与预测对数工资", xlabel="工作经验（年）", ylabel="预测对数工资")
ax.legend(frameon=False)
fig.tight_layout()
output1 = OUTPUT_DIR / "ch04-fig4-experience-curve.png"
fig.savefig(output1, dpi=220, bbox_inches="tight")

# 男女教育—工资预测线
educ_grid = np.linspace(df.educ.min(), df.educ.max(), 200)
fig, ax = plt.subplots(figsize=(8, 4.8))
for female_value, label, color in [(0, "男性", "#2A6F97"), (1, "女性", "#D17B88")]:
    pred = pd.DataFrame(
        {
            "const": 1.0,
            "educ": educ_grid,
            "exper": df.exper.mean(),
            "female": female_value,
            "exper2": df.exper.mean() ** 2,
            "educ_female": educ_grid * female_value,
        }
    )[m_inter.model.exog_names]
    ax.plot(educ_grid, m_inter.predict(pred), label=label, color=color, linewidth=2.5)
ax.set(title="教育回报的不同：一个交互项的直观表达", xlabel="教育年限", ylabel="预测对数工资")
ax.legend(frameon=False)
fig.tight_layout()
output2 = OUTPUT_DIR / "ch04-fig5-education-gender.png"
fig.savefig(output2, dpi=220, bbox_inches="tight")
print(f"\n图形已保存：{output1}\n图形已保存：{output2}")
