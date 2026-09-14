"""
第8章 工具变量模型 — 案例实践
配套教材：《计量经济学：理论与实践》

内容：生成模拟企业数据，依次完成四个任务——
1. OLS vs IV（内生性偏误诊断）
2. 第一阶段诊断（弱工具变量检测）
3. 多工具变量与过度识别检验
4. 内生性检验与稳健性汇总
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from linearmodels.iv import compare as iv_compare

# ============================================================
# 数据生成（500家企业，单一截面）
# ============================================================
np.random.seed(20250801)
n = 500

# 工具变量
ResDensity = np.random.normal(0, 1, n)                    # Z1: 科研资源密度
University = np.random.normal(0, 1, n) + 0.3 * ResDensity # Z2: 高校数量

# 不可观测的企业能力（内生性来源）
ability = np.random.normal(0, 1, n)

# 控制变量
size = np.clip(np.random.normal(5, 1, n), 2, None)   # 企业规模（对数）
age = np.random.uniform(1, 40, n)                     # 企业年龄

# 内生解释变量：研发投入
u_rd = np.random.normal(0, 1, n)
RD = (2.0 + 0.7*ResDensity + 0.1*University + 0.3*size
      - 0.02*age + 0.6*ability + u_rd)
RD = np.clip(RD, 0.1, None)

# 因变量：净资产收益率
v_roe = np.random.normal(0, 1.5, n)
ROE = (5.0 + 1.2*RD + 0.5*size - 0.01*age + 0.8*ability + v_roe)

df = pd.DataFrame({
    'ROE': ROE, 'RD': RD,
    'ResDensity': ResDensity, 'University': University,
    'size': size, 'age': age
})

print(f'样本量: {len(df)}')
print(df[['ROE', 'RD', 'ResDensity', 'University', 'size', 'age']].describe())

# ============================================================
# 任务1：OLS vs IV（内生性偏误诊断）
# ============================================================
print('\n' + '='*60)
print('任务1：OLS vs IV')
print('='*60)

# OLS（有偏——能力遗漏导致RD系数高估）
X_ols = sm.add_constant(df[['RD', 'size', 'age']])
m_ols = sm.OLS(df['ROE'], X_ols).fit(cov_type='HC1')

# 2SLS（用科研资源密度作为工具变量）
X_iv = sm.add_constant(df[['size', 'age']])
m_iv = IV2SLS(
    df['ROE'], X_iv,
    df['RD'], df['ResDensity']
).fit(cov_type='robust')

print('\n--- OLS vs IV 系数对比 ---')
comparison_1 = pd.DataFrame({
    'OLS': m_ols.params.round(5),
    'IV': m_iv.params.round(5),
})
print(comparison_1)

print('\n=== 提示 ===')
print('  OLS的RD系数 > IV的RD系数 → 能力偏误为正')
print('  ability同时与RD正相关、与ROE正相关 → OLS高估')
print('  在本DGP中，IV利用ResDensity推动的RD变动 → 应更接近设定值1.2')

# ============================================================
# 任务2：第一阶段诊断（弱工具变量检测）
# ============================================================
print('\n' + '='*60)
print('任务2：第一阶段诊断')
print('='*60)

# 第一阶段回归
X_stage1 = sm.add_constant(df[['ResDensity', 'size', 'age']])
m_stage1 = sm.OLS(df['RD'], X_stage1).fit(cov_type='HC1')
print('\n--- 第一阶段：RD = f(ResDensity, controls) ---')
print(m_stage1.summary().tables[1])

# 计算第一阶段F统计量（排除控制变量后ResDensity的偏F）
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 手动计算偏F：比较含Z和不含Z的第一阶段R²
X_stage1_no_z = sm.add_constant(df[['size', 'age']])
m_stage1_no_z = sm.OLS(df['RD'], X_stage1_no_z).fit()
r2_full = m_stage1.rsquared
r2_no_z = m_stage1_no_z.rsquared
n_obs = len(df)
k = 1  # 排除1个工具
# 完整模型包含常数、1个工具和2个控制变量，共4个参数
f_stat = ((r2_full - r2_no_z) / k) / ((1 - r2_full) / (n_obs - 4))
print(f'\n第一阶段偏F统计量（ResDensity）: {f_stat:.2f}')

print('\n=== 提示 ===')
print('  常见的F>10只是简单同方差设定下的经验预警线，不是“强工具证书”')
print('  F较小提示弱识别风险；正式研究应报告适合模型与误差结构的诊断')
print('  本DGP中ResDensity系数约0.7，因此第一阶段应当较强')

# 弱工具对比演示
noise_iv = np.random.normal(0, 1, n)
X_weak = sm.add_constant(pd.DataFrame({'noise': noise_iv, 'size': size, 'age': age}))
m_weak = sm.OLS(RD, X_weak).fit()
r2_weak = m_weak.rsquared
f_weak = ((r2_weak - r2_no_z) / k) / ((1 - r2_weak) / (n_obs - 4))
print(f'\n弱工具对比——纯噪声作为IV的偏F: {f_weak:.2f} （演示弱工具）')

# ============================================================
# 任务3：多工具变量与过度识别检验
# ============================================================
print('\n' + '='*60)
print('任务3：多工具与过度识别')
print('='*60)

# 使用两个工具变量
m_iv2 = IV2SLS(
    df['ROE'], X_iv,
    df['RD'], df[['ResDensity', 'University']]
).fit(cov_type='robust')

print('\n--- 单IV vs 双IV 系数对比 ---')
comparison_3 = pd.DataFrame({
    '单IV (Z1)': m_iv.params.round(5),
    '双IV (Z1+Z2)': m_iv2.params.round(5),
})
print(comparison_3)

# 过度识别检验（稳健的Wooldridge score test）
overid = m_iv2.wooldridge_overid
print('\n--- 过度识别检验（Wooldridge稳健得分检验） ---')
print(f'检验统计量: {overid.stat:.4f}')
print(f'p值:        {overid.pval:.4f}')

print('\n=== 提示 ===')
print('  H0: 过度识别约束成立（解释依赖至少有足够有效工具等前提）')
print('  p > 0.05 → 不拒绝H0，但不能证明所有工具都外生')
print('  p < 0.05 → 拒绝H0 → 至少有一个工具是内生的')
print('  恰好识别时（1个IV）无法做此检验')
print('  本DGP中两个IV都外生 → 预期不拒绝H0')

# ============================================================
# 任务4：内生性检验与稳健性汇总
# ============================================================
print('\n' + '='*60)
print('任务4：内生性检验与汇总')
print('='*60)

# Durbin-Wu-Hausman 内生性检验（比较OLS和IV）
# 通过将第一阶段残差加入OLS方程实现
# 第一阶段残差
rd_pred = m_stage1.predict(X_stage1)
rd_resid = df['RD'] - rd_pred

# 第二阶段加入残差
X_dwh = sm.add_constant(pd.DataFrame({
    'RD': df['RD'], 'size': df['size'], 'age': df['age'],
    'resid': rd_resid
}))
m_dwh = sm.OLS(df['ROE'], X_dwh).fit(cov_type='HC1')
dwh_t = m_dwh.tvalues['resid']
dwh_pval = m_dwh.pvalues['resid']
print(f'\n--- Durbin-Wu-Hausman 内生性检验 ---')
print(f'  残差项 t值: {dwh_t:.4f}')
print(f'  残差项 p值: {dwh_pval:.4f}')

print('\n=== 提示 ===')
print('  H0: RD是外生的（OLS和IV都一致）')
print('  p < 0.05 → 数据反对RD外生；是否采用IV还取决于工具是否可信')
print('  p > 0.05 → 未发现明确内生性证据，不等于证明RD外生')

# 最终汇总
print('\n' + '='*60)
print('最终汇总：三模型对比')
print('='*60)

summary = pd.DataFrame({
    'OLS': m_ols.params[['RD', 'size', 'age']].round(5),
    'IV (Z1)': m_iv.params[['RD', 'size', 'age']].round(5),
    'IV (Z1+Z2)': m_iv2.params[['RD', 'size', 'age']].round(5),
})
print(summary)

print('\n=== IV分析标准流程 ===')
print('  步骤1: OLS基准 → 确认存在内生性偏误的可能')
print('  步骤2: 第一阶段诊断 → 评估弱识别风险')
print('  步骤3: 2SLS估计 → 在识别假设成立时解释局部因果效应')
print('  步骤4: 过度识别/DWH检验 → 提供诊断证据，不替代制度论证')
