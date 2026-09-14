/*══════════════════════════════════════════════════════════════
  第6章 离散选择模型 — 案例实践
  配套教材：《计量经济学：理论与实践》
  功能：生成模拟满意度调查数据，依次完成四个任务：
        任务1：二元选择模型（LPM / Logit / Probit）
        任务2：边际效应计算与比较
        任务3：有序Probit模型
        任务4：四模型汇总比较
  ══════════════════════════════════════════════════════════════*/

clear
set more off
set seed 20250710
set obs 1000

* ============================================================
* 数据生成
* ============================================================
gen income  = max(1, min(50, rnormal(10, 5)))
gen edu     = max(6, min(22, round(rnormal(12, 3))))
gen age     = max(22, min(65, round(rnormal(40, 12))))
gen married = rbinomial(1, 0.75)
gen health  = max(1, min(5, round(rnormal(3, 1))))
gen urban   = rbinomial(1, 0.55)
gen female  = rbinomial(1, 0.5)

* 潜在满意度（Probit DGP：正态误差）
gen u = rnormal(0, 1)
gen y_star = -1.0 + 0.05*income + 0.08*edu - 0.005*age ///
             + 0.3*married + 0.25*health + 0.1*urban - 0.05*female + u

* 二元因变量：是否满意
gen happy = (y_star > 0)

* 有序因变量：五等级满意度（阈值 -1.5, -0.5, 0.5, 1.5）
gen satisfaction = 1 if y_star <= -1.5
replace satisfaction = 2 if y_star > -1.5 & y_star <= -0.5
replace satisfaction = 3 if y_star > -0.5 & y_star <= 0.5
replace satisfaction = 4 if y_star > 0.5 & y_star <= 1.5
replace satisfaction = 5 if y_star > 1.5 & y_star < .

label variable happy        "是否满意（1=满意）"
label variable satisfaction "满意度等级（1-5）"
label variable income       "家庭年收入（万元）"
label variable edu          "受教育年限"
label variable age          "年龄"
label variable married      "已婚（=1）"
label variable health       "自评健康（1-5）"
label variable urban        "城镇户口（=1）"
label variable female       "女性（=1）"

* ============================================================
* 任务1：二元选择模型（LPM / Logit / Probit）
* ============================================================
display _n "========== 任务1：LPM vs Logit vs Probit =========="

local rhs "income edu age i.married health i.urban i.female"

* LPM（稳健标准误）
reg happy `rhs', robust
estimates store m_lpm

* 检查预测值范围
predict lpm_pred
count if lpm_pred < 0 | lpm_pred > 1
local n_out = r(N)
local n_total = _N
display _n "LPM 预测值落在 [0,1] 之外的观测数: " `n_out' " / " _N
display "  比例: " %5.2f 100*`n_out'/`n_total' "%"
drop lpm_pred

* Logit
logit happy `rhs'
estimates store m_logit

* Probit
probit happy `rhs'
estimates store m_probit

* 三模型对比表
estimates table m_lpm m_logit m_probit, ///
    stats(N) b(%9.4f) se(%9.4f) drop(_cons)

display _n "=== 提示 ==="
display "  Logit 系数 ≈ Probit 系数的 1.6–1.8 倍"
display "  两者方向、显著性应一致"
display "  LPM 系数 ≈ Logit/Probit 的 AME（而非原始系数）"

* ============================================================
* 任务2：边际效应（AME）
* ============================================================
display _n "========== 任务2：边际效应计算与比较 =========="

* Logit 平均边际效应
logit happy `rhs'
estimates store m_logit
margins, dydx(*) post
estimates store m_logit_ame

* Probit 平均边际效应
probit happy `rhs'
estimates store m_probit
margins, dydx(*) post
estimates store m_probit_ame

* AME 对比表
estimates table m_logit_ame m_probit_ame, ///
    stats(N) b(%9.5f) se(%9.5f)

display _n "=== 提示 ==="
display "  1. 两列 AME 应非常接近"
display "  2. AME 直接解释为：X增加1单位，概率平均变化XX个百分点"
display "  3. 这就是 Logit/Probit 的 '翻译' —— 转回概率语言"

* ============================================================
* 任务3：有序Probit模型
* ============================================================
display _n "========== 任务3：有序Probit模型 =========="

oprobit satisfaction `rhs'
estimates store m_oprobit

* 各类别的边际效应
display _n "--- 各类别边际效应 ---"

forvalues j = 1/5 {
    display _n "类别 `j' 的 AME:"
    margins, dydx(*) predict(outcome(`j'))
}

display _n "=== 提示 ==="
display "  1. 类别1（非常不满意）：AME 应为负（收入↑ → 概率↓）"
display "  2. 类别5（非常满意）：AME 应为正（收入↑ → 概率↑）"
display "  3. 中间类别：符号和大小取决于流入与流出，不能预设为接近零"
display "  4. 各类别AME之和应在数值精度内接近零"

* ============================================================
* 任务4：四模型汇总比较
* ============================================================
display _n "========== 任务4：四模型汇总比较 =========="

estimates restore m_lpm
estimates store m1
estimates restore m_logit
estimates store m2
estimates restore m_probit
estimates store m3
estimates restore m_oprobit
estimates store m4

estimates table m1 m2 m3 m4, ///
    stats(N) b(%9.4f) se(%9.4f) drop(_cons)

display _n "=== 模型选择提示 ==="
display "  LPM     → 系数≈概率变化，最直观，但预测值可能超界"
display "  Logit   → 可报告机会比（or选项），医学/营销常用"
display "  Probit  → 潜变量框架，经济学常用"
display "  有序Probit → 因变量为有序等级时的首选"
display ""
display "  报告方式应由因变量类型、研究问题和读者需求决定；"
display "  无论选择哪种非线性模型，都应给出可解释的概率效应。"
