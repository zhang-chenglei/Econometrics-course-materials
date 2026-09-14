/*══════════════════════════════════════════════════════════════
  第7章 面板数据模型 — 案例实践
  配套教材：《计量经济学：理论与实践》
  功能：生成模拟上市公司面板数据，依次完成四个任务：
        任务1：混合OLS与固定效应对比
        任务2：三种FE方法验证（组内去心 / LSDV / 一阶差分）
        任务3：随机效应与豪斯曼检验
        任务4：双向固定效应
  ══════════════════════════════════════════════════════════════*/

clear
set more off
set seed 20250715

* ============================================================
* 数据生成（200家公司 × 5年 = 1000观测值）
* ============================================================
set obs 200
gen company_id = _n

* 公司固定效应 α_i（与R&D正相关——创新文化强的公司R&D高、利润也高）
gen alpha_i = rnormal(0, 1.5)

* 扩展为面板：每家公司5年
expand 5
bysort company_id: gen year = _n
xtset company_id year

* 年份固定效应 γ_t（模拟宏观趋势：逐年小幅上升）
gen gamma_t = -0.3 if year == 1
replace gamma_t = -0.15 if year == 2
replace gamma_t = 0 if year == 3
replace gamma_t = 0.15 if year == 4
replace gamma_t = 0.3 if year == 5

* 时变解释变量
* R&D投入：与公司FE正相关（创新文化→高R&D）
gen RD = rnormal(5, 2) + 0.3 * alpha_i
replace RD = max(0.5, RD)

* 总资产：与公司FE弱正相关
gen assets = rnormal(100, 30) + 0.15 * alpha_i
replace assets = max(10, assets)

* 负债率：独立的时变变量（与控制变量）
gen debt = rnormal(30, 10)
replace debt = max(0, debt)

* 因变量：利润
* DGP: profit = 2 + 0.18*RD + 0.04*assets - 0.03*debt + alpha_i + gamma_t + u
gen u = rnormal(0, 2)
gen profit = 2.0 + 0.18*RD + 0.04*assets - 0.03*debt + alpha_i + gamma_t + u

label variable company_id "公司编号"
label variable year       "年份"
label variable RD         "研发投入（百万元）"
label variable assets     "总资产（百万元）"
label variable debt       "负债率（%）"
label variable profit     "净利润（百万元）"

display _n "样本量: " _N
display "公司数: " 200 ", 年份: 5"
display "面板结构: 平衡面板 (200 × 5)"
sum profit RD assets debt

* ============================================================
* 任务1：混合OLS与固定效应对比
* ============================================================
display _n "========== 任务1：混合OLS vs FE =========="

* 混合OLS（忽略面板结构，稳健标准误）
reg profit RD assets debt, vce(cluster company_id)
estimates store m_pooled

* 固定效应（组内去心法）
xtreg profit RD assets debt, fe vce(cluster company_id)
estimates store m_fe

* 对比表
display _n "--- 混合OLS vs FE 系数对比 ---"
estimates table m_pooled m_fe, ///
    stats(N r2) b(%9.4f) se(%9.4f) drop(_cons)

display _n "=== 提示 ==="
display "  如果混合OLS的RD系数 > FE的RD系数："
display "  → 公司FE与RD正相关，混合OLS高估了R&D的真实效应"
display "  → FE通过组内去心消除了不变的公司特质"
display "  → FE的估计更接近R&D的'净效应'"

* ============================================================
* 任务2：三种FE方法验证
* ============================================================
display _n "========== 任务2：三种FE方法对比 =========="

* 方法1：组内去心法（xtreg, fe）
xtreg profit RD assets debt, fe
estimates store m_within
display "  组内去心法 R&D系数: " %9.4f _b[RD]

* 方法2：LSDV（加公司虚拟变量）
reg profit RD assets debt i.company_id
estimates store m_lsdv
display "  LSDV法 R&D系数: " %9.4f _b[RD]

* 方法3：一阶差分法（手动差分）
sort company_id year
foreach var in profit RD assets debt {
    by company_id: gen d_`var' = `var' - `var'[_n-1]
}
reg d_profit d_RD d_assets d_debt, nocons
estimates store m_fd
display "  一阶差分法 R&D系数: " %9.4f _b[d_RD]

* 三种方法对比表
display _n "--- 三种FE方法 R&D系数对比 ---"
estimates table m_within m_lsdv m_fd, ///
    stats(N) b(%9.4f) se(%9.4f) keep(RD d_RD)

display _n "=== 提示 ==="
display "  组内去心法 & LSDV → 系数应完全一致（等价变换）"
display "  一阶差分法 → 系数通常接近但不完全一致"
display "    → 差分法丢失第一期数据（N减少）"
display "    → 差分法对测量误差更敏感"

* 清理差分变量
drop d_*

* ============================================================
* 任务3：随机效应与豪斯曼检验
* ============================================================
display _n "========== 任务3：随机效应与豪斯曼检验 =========="

* 随机效应模型（主结果使用企业聚类标准误）
xtreg profit RD assets debt, re vce(cluster company_id)
estimates store m_re
display "  随机效应 R&D系数: " %9.4f _b[RD]

* FE vs RE 系数对比
display _n "--- FE vs RE 系数对比 ---"
estimates table m_fe m_re, ///
    stats(N) b(%9.4f) se(%9.4f) keep(RD assets debt)

* 豪斯曼检验：另估计传统协方差版本，避免稳健协方差不兼容
display _n "--- 豪斯曼检验 ---"
xtreg profit RD assets debt, fe
estimates store m_fe_haus
xtreg profit RD assets debt, re
estimates store m_re_haus
hausman m_fe_haus m_re_haus, sigmamore

display _n "=== 提示 ==="
display "  H0: 随机效应一致（RE假设成立，α_i与解释变量不相关）"
display "  H1: 固定效应一致，随机效应不一致"
display "  如果 p < 0.05 → 拒绝RE → 使用FE"
display "  本DGP中α_i与RD正相关 → 预期拒绝RE"
display ""
display "  即使不拒绝RE，也应结合研究目标和相关性假设，而不是机械选RE"

* ============================================================
* 任务4：双向固定效应
* ============================================================
display _n "========== 任务4：双向固定效应 =========="

* 双向固定效应（公司FE + 年份FE）
xtreg profit RD assets debt i.year, fe vce(cluster company_id)
estimates store m_twoway

* 三模型对比：Pooled OLS vs 单向FE vs 双向FE
display _n "--- 混合OLS vs 单向FE vs 双向FE ---"
estimates table m_pooled m_fe m_twoway, ///
    stats(N r2) b(%9.4f) se(%9.4f) keep(RD assets debt)

* 年份FE的联合显著性
testparm i.year

display _n "=== 提示 ==="
display "  1. 比较单向FE和双向FE的R&D系数："
display "     → 如果接近 → 年份冲击与R&D相关性不大"
display "     → 如果差异明显 → 省略年份FE会导致额外偏误"
display "  2. 查看年份虚拟变量系数："
display "     → 它们捕捉了各年的全行业共性冲击"
display "     → 显著年份FE = 该年存在明显的宏观扰动"
display "  3. 双向FE是面板论文中最常见的基准设定"

* ============================================================
* 最终汇总
* ============================================================
display _n "========== 最终汇总：四任务对比表 =========="
estimates table m_pooled m_fe m_re m_twoway, ///
    stats(N r2) b(%9.4f) se(%9.4f) keep(RD assets debt)

display _n "=== 面板数据分析标准流程 ==="
display "  步骤1: xtset → 声明面板结构（不是可选的！）"
display "  步骤2: 比较Pooled OLS、FE与RE的假设和估计对象"
display "  步骤3: Hausman检验 → 提供关于FE/RE差异的统计证据"
display "  步骤4: 加入时间FE → 双向固定效应"
display ""
display "  这个顺序不是随意的——每一步都有明确的逻辑目的。"
display "  记不住命令没关系，但必须记住这个顺序和意义。"
