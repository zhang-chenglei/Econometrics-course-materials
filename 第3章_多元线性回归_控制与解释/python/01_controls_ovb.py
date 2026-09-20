"""案例二：AI 使用与课程成绩——控制变量与遗漏变量偏误。

生成一张模型比较表，以及一张图（遗漏模型与完整模型的 AI 系数及其置信区间），
全部保存到本脚本同级的 output/ 文件夹。

教学用模拟数据，数据生成过程人为设定且已知：

    low_bg = 1{family_bg < 0}        家庭背景较弱=1，家庭背景较好=0（基准组）
    ai    = 3 + 3.2*ability + 0.25*age - 1.2*low_bg + v,  v ~ N(0, 3.5)
    score = 56 + 1.20*ai + 4.5*ability + 0.6*age - 1.8*low_bg + u,  u ~ N(0, 5.5)

其中 ability 是认知能力，它同时影响 AI 使用时间和课程成绩，是本案例要考察的
遗漏变量。真实系数 1.20 是我们希望估计出来的目标参数。全部为教学用合成数据，
不代表现实效果。
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 随机种子：Python 与 Stata 的随机数算法不同，同一份数据生成过程也会抽到不同的
# 样本。这里各自固定一个种子，使两种软件的输出都落在真实系数附近，便于课堂对照；
# 数值不必完全一致，图形形状和结论一致即可。
SEED = 20260914
N = 800
TRUE_BETA_AI = 1.20

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_controls_ovb.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

ability = rng.normal(0, 1, size=N)                                # 认知能力
age = np.clip(np.round(rng.normal(20, 1.6, size=N)), 17, 26)      # 年龄
family_bg = rng.normal(0, 1, size=N)                              # 家庭背景因子
low_bg = (family_bg < 0).astype(int)                              # 家庭背景较弱=1，较好=0（基准组）

ai = (
    3 + 3.2 * ability + 0.25 * age - 1.2 * low_bg + rng.normal(0, 3.5, size=N)
)
ai = np.clip(ai, 0, 30)
score = (
    56
    + TRUE_BETA_AI * ai
    + 4.5 * ability
    + 0.6 * age
    - 1.8 * low_bg
    + rng.normal(0, 5.5, size=N)
)

df = pd.DataFrame({"score": score, "ai": ai, "age": age, "low_bg": low_bg, "ability": ability})

# 两个模型：遗漏模型只放核心变量，完整模型加入年龄、家庭背景和认知能力
specs = [
    ("(1) 只含 AI", ["ai"]),
    ("(2) 加年龄、家庭背景、认知能力", ["ai", "age", "low_bg", "ability"]),
]
fits = {}
rows = []
for label, xs in specs:
    fit = sm.OLS(df["score"], sm.add_constant(df[xs])).fit()
    fits[label] = fit
    ci = fit.conf_int().loc["ai"]
    rows.append(
        {
            "model": label,
            "ai_coef": fit.params["ai"],
            "ai_se": fit.bse["ai"],
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "r_squared": fit.rsquared,
        }
    )

comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_DIR / "ch03_model_comparison.csv", index=False, encoding="utf-8-sig")

print("模型比较（被解释变量：课程成绩；真实 AI 系数 = 1.20）：")
print(comparison.round(4).to_string(index=False))
print("\n变量相关系数：")
print(df[["score", "ai", "ability", "age", "low_bg"]].corr().round(3).to_string())
print("\n完整模型摘要：")
full = fits["(2) 加年龄、家庭背景、认知能力"]
print(full.summary().tables[1])
print(f"\nAI系数t检验：t={full.tvalues['ai']:.3f}, p={full.pvalues['ai']:.4g}")
joint = full.f_test("age = 0, low_bg = 0")
print(f"年龄与家庭背景联合检验：F={float(joint.fvalue):.3f}, p={float(joint.pvalue):.4g}")

# 图3-4：AI 系数在遗漏模型与完整模型中的位置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

y = np.arange(len(comparison))[::-1]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.errorbar(
    comparison["ai_coef"],
    y,
    xerr=[
        comparison["ai_coef"] - comparison["ci_lower"],
        comparison["ci_upper"] - comparison["ai_coef"],
    ],
    fmt="o",
    color="#2A6F97",
    ecolor="#6C8EAD",
    capsize=4,
    markersize=7,
)
ax.axvline(TRUE_BETA_AI, color="#C44536", linestyle="--", linewidth=1.6,
           label=f"真实系数 = {TRUE_BETA_AI}")
ax.set_yticks(y, comparison["model"])
ax.set_xlabel("AI 使用时间的系数及 95% 置信区间")
ax.set_title("遗漏变量被控制后，AI 系数回到真实值附近")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
output = OUTPUT_DIR / "ch03-fig4-ovb-coefficient-path.png"
fig.savefig(output, dpi=220, bbox_inches="tight")
print(f"\n图形已保存：{output}")
