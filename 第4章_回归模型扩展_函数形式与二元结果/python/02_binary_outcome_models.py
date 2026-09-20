"""第4章精简二元结果案例：LPM、Logit与Probit。

沿用第4、5章的固定教学样本，按事先确定的75分标准定义
``pass = 1(score >= 75)``。本脚本只比较三种二元结果模型，
不扩展到有序选择模型。

输出（本脚本同级 output/）：
- ch04_binary_model_summary.csv       原始系数、标准误与预测范围
- ch04_binary_marginal_effects.csv    AI使用时间的概率边际效应
- ch04_binary_prediction_check.csv    三种模型的预测越界检查
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260915
N = 800
PASS_THRESHOLD = 75

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
ability = rng.normal(0, 1, size=N)
age = np.clip(np.round(rng.normal(20, 1.6, size=N)), 17, 26)
strong = (ability > 0).astype(int)
family_bg = rng.normal(0, 1, size=N)
parent_edu = 12 + 2.2 * family_bg + rng.normal(0, 0.30, size=N)
family_income = 8 + 1.9 * family_bg + rng.normal(0, 0.30, size=N)
ai = 3 + 3.2 * ability + 0.25 * age + rng.normal(0, 3.5, size=N)
ai = np.clip(ai, 0, 30)
ai2 = ai**2
ai_strong = ai * strong
sigma = 2.0 + 0.35 * ai
score = (
    56
    + 1.20 * ai
    - 0.060 * ai2
    + 0.55 * ai_strong
    + 4.5 * ability
    + 0.6 * age
    + 0.20 * parent_edu
    + 0.15 * family_income
    + rng.normal(0, sigma, size=N)
)

df = pd.DataFrame({"pass": (score >= PASS_THRESHOLD).astype(int), "ai": ai, "age": age, "strong": strong})
X = sm.add_constant(df[["ai", "age", "strong"]])

lpm = sm.OLS(df["pass"], X).fit(cov_type="HC1")
logit = sm.Logit(df["pass"], X).fit(disp=False, cov_type="HC1")
probit = sm.Probit(df["pass"], X).fit(disp=False, cov_type="HC1")

predictions = {
    "LPM": lpm.predict(X),
    "Logit": logit.predict(X),
    "Probit": probit.predict(X),
}

model_rows = []
for name, model in (("LPM", lpm), ("Logit", logit), ("Probit", probit)):
    pred = predictions[name]
    model_rows.append(
        {
            "model": name,
            "ai_raw_coefficient": model.params["ai"],
            "ai_robust_se_HC1": model.bse["ai"],
            "predicted_probability_min": float(pred.min()),
            "predicted_probability_max": float(pred.max()),
        }
    )
model_summary = pd.DataFrame(model_rows)
model_summary.to_csv(OUTPUT_DIR / "ch04_binary_model_summary.csv", index=False, encoding="utf-8-sig")

logit_me = logit.get_margeff(at="overall", method="dydx", dummy=True).summary_frame().loc["ai"]
probit_me = probit.get_margeff(at="overall", method="dydx", dummy=True).summary_frame().loc["ai"]
marginal_effects = pd.DataFrame(
    [
        {
            "model": "LPM",
            "estimand": "AI系数（概率点平均变化）",
            "ai_marginal_effect": lpm.params["ai"],
            "std_err": lpm.bse["ai"],
            "ci_lower": lpm.conf_int().loc["ai", 0],
            "ci_upper": lpm.conf_int().loc["ai", 1],
        },
        {
            "model": "Logit",
            "estimand": "AI导数型AME",
            "ai_marginal_effect": logit_me["dy/dx"],
            "std_err": logit_me["Std. Err."],
            "ci_lower": logit_me["Conf. Int. Low"],
            "ci_upper": logit_me["Cont. Int. Hi."],
        },
        {
            "model": "Probit",
            "estimand": "AI导数型AME",
            "ai_marginal_effect": probit_me["dy/dx"],
            "std_err": probit_me["Std. Err."],
            "ci_lower": probit_me["Conf. Int. Low"],
            "ci_upper": probit_me["Cont. Int. Hi."],
        },
    ]
)
marginal_effects.to_csv(
    OUTPUT_DIR / "ch04_binary_marginal_effects.csv", index=False, encoding="utf-8-sig"
)

prediction_check = pd.DataFrame(
    [
        {
            "model": name,
            "minimum": float(pred.min()),
            "maximum": float(pred.max()),
            "below_zero": int((pred < 0).sum()),
            "above_one": int((pred > 1).sum()),
        }
        for name, pred in predictions.items()
    ]
)
prediction_check.to_csv(
    OUTPUT_DIR / "ch04_binary_prediction_check.csv", index=False, encoding="utf-8-sig"
)

print(f"达标线：{PASS_THRESHOLD}分；样本达标率：{df['pass'].mean():.3f}")
print("\n模型概要：")
print(model_summary.round(5).to_string(index=False))
print("\nAI使用时间的概率边际效应：")
print(marginal_effects.round(5).to_string(index=False))
print("\n预测范围检查：")
print(prediction_check.round(5).to_string(index=False))
