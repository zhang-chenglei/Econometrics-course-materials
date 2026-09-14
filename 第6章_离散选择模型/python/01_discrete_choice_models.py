"""
第6章 离散选择模型 — 案例实践
配套教材：《计量经济学：理论与实践》

内容：生成模拟满意度调查数据，依次完成四个任务——
1. 二元选择模型（LPM / Logit / Probit）
2. 边际效应计算与比较
3. 有序Probit模型
4. 四模型汇总比较
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ============================================================
# 数据生成
# ============================================================
np.random.seed(20250710)
n = 1000

income  = np.clip(np.random.normal(10, 5, size=n), 1, 50)
edu     = np.clip(np.round(np.random.normal(12, 3, size=n)), 6, 22).astype(int)
age     = np.clip(np.round(np.random.normal(40, 12, size=n)), 22, 65).astype(int)
married = np.random.binomial(1, 0.75, size=n)
health  = np.clip(np.round(np.random.normal(3, 1, size=n)), 1, 5).astype(int)
urban   = np.random.binomial(1, 0.55, size=n)
female  = np.random.binomial(1, 0.5, size=n)

# 潜在满意度（Probit DGP：正态误差）
u = np.random.normal(0, 1, size=n)
y_star = (-1.0 + 0.05*income + 0.08*edu - 0.005*age
          + 0.3*married + 0.25*health + 0.1*urban - 0.05*female + u)

# 二元因变量
happy = (y_star > 0).astype(int)

# 有序因变量（五等级，阈值 -1.5, -0.5, 0.5, 1.5）
satisfaction = np.select(
    [y_star <= -1.5, y_star <= -0.5, y_star <= 0.5, y_star <= 1.5, y_star > 1.5],
    [1, 2, 3, 4, 5]
)

df = pd.DataFrame({
    'happy': happy, 'satisfaction': satisfaction,
    'income': income, 'edu': edu, 'age': age,
    'married': married, 'health': health,
    'urban': urban, 'female': female
})

vars_list = ['income', 'edu', 'age', 'married', 'health', 'urban', 'female']
X = sm.add_constant(df[vars_list])

print(f'样本量: {len(df)}')
print(f'happy=1 比例: {happy.mean():.3f}')
print(f'satisfaction 分布:\n{df["satisfaction"].value_counts().sort_index()}')
print(df.head())

# ============================================================
# 任务1：二元选择模型（LPM / Logit / Probit）
# ============================================================
print('\n' + '='*60)
print('任务1：LPM vs Logit vs Probit')
print('='*60)

# LPM（稳健标准误）
m_lpm = sm.OLS(df['happy'], X).fit(cov_type='HC1')

# 检查预测值范围
lpm_pred = m_lpm.predict(X)
n_out = np.sum((lpm_pred < 0) | (lpm_pred > 1))
print(f'\nLPM 预测值落在 [0,1] 之外的观测数: {n_out} / {n}')
print(f'  比例: {100*n_out/n:.2f}%')

# Logit
m_logit = Logit(df['happy'], X).fit(disp=False)

# Probit
m_probit = Probit(df['happy'], X).fit(disp=False)

print('\n--- 三模型系数对比 ---')
comparison_1 = pd.DataFrame({
    'LPM': m_lpm.params,
    'Logit': m_logit.params,
    'Probit': m_probit.params,
    'Logit/Probit': m_logit.params / m_probit.params
}).round(5)
print(comparison_1)

print('\n=== 提示 ===')
print('  Logit 系数 ≈ Probit 系数的 1.6–1.8 倍')
print('  两者方向、显著性应一致')
print('  LPM 系数 ≈ Logit/Probit 的 AME（而非原始系数）')

# ============================================================
# 任务2：边际效应（AME）
# ============================================================
print('\n' + '='*60)
print('任务2：边际效应计算与比较')
print('='*60)

# Logit AME
logit_ame = m_logit.get_margeff(at='overall', dummy=True)
print('\n--- Logit 平均边际效应 ---')
print(logit_ame.summary())

# Probit AME
probit_ame = m_probit.get_margeff(at='overall', dummy=True)
print('\n--- Probit 平均边际效应 ---')
print(probit_ame.summary())

print('\n=== 提示 ===')
print('  1. 两列 AME 应非常接近')
print('  2. 连续变量按增加1单位解释；二元变量按从0变为1的离散变化解释')
print('  3. 这就是 Logit/Probit 的"翻译"——转回概率语言')

# ============================================================
# 任务3：有序Probit模型
# ============================================================
print('\n' + '='*60)
print('任务3：有序Probit模型')
print('='*60)

# 有序Probit
m_oprobit = OrderedModel(df['satisfaction'], df[vars_list],
                         distr='probit').fit(method='bfgs', disp=False)
print('\n--- 有序Probit 系数 ---')
print(m_oprobit.summary())

# 各类别边际效应（手动计算）
print('\n--- 各类别边际效应（收入变量的AME）---')
# 用数值微分近似
h = 0.001
income_idx = list(vars_list).index('income')
for j in range(1, 6):
    # 原始预测概率
    pred_orig = m_oprobit.predict()
    prob_j = pred_orig.iloc[:, j-1].mean() if hasattr(pred_orig, 'iloc') else pred_orig[:, j-1].mean()

    # 实际边际效应需要用数值微分，此处展示近似思路
    # 在 statsmodels 中 get_margeff 对 OrderedModel 支持有限
    # 下面采用简单的数值方法
    X_plus = df[vars_list].copy()
    X_plus.iloc[:, income_idx] += h
    pred_plus = m_oprobit.predict(X_plus)
    prob_j_plus = pred_plus.iloc[:, j-1].mean() if hasattr(pred_plus, 'iloc') else pred_plus[:, j-1].mean()

    ame_j = (prob_j_plus - prob_j) / h
    print(f'  类别{j} AME: {ame_j:.5f}')

print('\n=== 提示 ===')
print('  1. 类别1（非常不满意）：AME 应为负（收入↑ → 概率↓）')
print('  2. 类别5（非常满意）：AME 应为正（收入↑ → 概率↑）')
print('  3. 中间类别：符号和大小取决于流入与流出，不能预设为接近零')
print('  4. 五个类别的 AME 之和应在数值精度内接近零')

# ============================================================
# 任务4：四模型汇总比较
# ============================================================
print('\n' + '='*60)
print('任务4：四模型汇总比较')
print('='*60)

summary_4 = pd.DataFrame({
    'LPM系数': m_lpm.params,
    'Logit系数': m_logit.params,
    'Probit系数': m_probit.params,
    '有序Probit系数': m_oprobit.params,
}).round(5)
print(summary_4)

print('\n=== 模型选择提示 ===')
print('  LPM       → 系数≈概率变化，最直观，但预测值可能超界')
print('  Logit     → 可报告机会比（exp(coef)），医学/营销常用')
print('  Probit    → 潜变量框架，经济学常用')
print('  有序Probit → 因变量为有序等级时的首选')
print()
print('  报告方式应由因变量类型、研究问题和读者需求决定；')
print('  无论选择哪种非线性模型，都应给出可解释的概率效应。')
