"""引例：AI 使用时间与课程成绩。

生成三张图（散点图、加入拟合线、总体回归线与样本回归线）和一份样本数据表，
全部保存到本脚本同级的 output/ 文件夹。

教学用合成数据，数据生成过程人为设定且已知：
    score_i = 70 + 1.2 * ai_i + u_i,  u ~ N(0, 5.5)
其中 ai 为每周用于学习的生成式 AI 时间（小时）。案例结论只用于说明回归线在
做什么，不代表现实中的学习效果。
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260914
N = 200
BETA0, BETA1 = 70.0, 1.2
SIGMA = 5.5
AI_MAX = 15.0

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 00_ai_score_scatter.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
ai_time = rng.uniform(0, AI_MAX, size=N)
score = BETA0 + BETA1 * ai_time + rng.normal(0, SIGMA, size=N)
df = pd.DataFrame({"ai_time": ai_time, "score": score})
df.to_csv(OUTPUT_DIR / "ch01_ai_score_sample.csv", index=False, encoding="utf-8-sig")

fit = sm.OLS(df["score"], sm.add_constant(df["ai_time"])).fit()
print(f"样本量：{N}")
print(f"总体回归线：score = {BETA0:.0f} + {BETA1:.1f} * ai")
print(f"样本回归线：score = {fit.params['const']:.3f} + {fit.params['ai_time']:.3f} * ai")
print(f"样本相关系数 r = {df['ai_time'].corr(df['score']):.4f}")
print(f"R² = {fit.rsquared:.4f}")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

x_grid = np.linspace(0, AI_MAX, 200)
sample_line = fit.params["const"] + fit.params["ai_time"] * x_grid
population_line = BETA0 + BETA1 * x_grid

# 图1-1 散点图
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.ai_time, df.score, s=22, color="#2A6F97", alpha=0.72)
ax.set(xlabel="每周用于学习的 AI 时间（小时）", ylabel="课程成绩（分）")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig1-ai-score-scatter.png", dpi=220, bbox_inches="tight")

# 图1-2 拟合图
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.ai_time, df.score, s=22, color="#2A6F97", alpha=0.72)
ax.plot(x_grid, sample_line, color="#C44536", linewidth=2.2, label="样本回归线")
ax.set(xlabel="每周用于学习的 AI 时间（小时）", ylabel="课程成绩（分）")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig2-ai-score-fit.png", dpi=220, bbox_inches="tight")

# 图1-4 总体回归线与样本回归线
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(df.ai_time, df.score, s=18, color="#999999", alpha=0.55, label="样本点")
ax.plot(x_grid, sample_line, color="#2A6F97", linewidth=2.2, label="样本回归线")
ax.plot(
    x_grid, population_line, color="#C44536", linewidth=2.2,
    linestyle="--", label="总体回归线",
)
ax.set(xlabel="每周用于学习的 AI 时间（小时）", ylabel="课程成绩（分）")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig4-population-vs-sample.png", dpi=220, bbox_inches="tight")

print(f"\n图形已保存至：{OUTPUT_DIR}")
