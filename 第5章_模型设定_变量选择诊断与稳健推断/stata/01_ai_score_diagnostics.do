* 第5章案例：AI 使用与课程成绩——模型设定与诊断
* 沿用第4章案例的同一份数据（同一生成过程）。
* 第4章的模型形式分析见 ch04/01_ai_score_form_specification.do。
* Stata 与 Python 的随机数发生器不同，两边样本不会逐个相同，但数据特征与结论一致。

clear all
set more off
set seed 20260915
capture mkdir output
set obs 800

* —— 与第4章共用的一份数据 ——
gen ability = rnormal()
gen age = min(max(round(rnormal(20,1.6)),17),26)
gen strong = ability > 0
gen family_bg = rnormal()
gen parent_edu = 12 + 2.2*family_bg + rnormal(0,0.30)
gen family_income = 8 + 1.9*family_bg + rnormal(0,0.30)
gen ai = min(max(3 + 3.2*ability + 0.25*age + rnormal(0,3.5),0),30)
gen ai2 = ai^2
gen ai_strong = ai*strong
gen sigma = 2.0 + 0.35*ai
gen u = rnormal(0,sigma)
gen score = 56 + 1.20*ai - 0.060*ai2 + 0.55*ai_strong + 4.5*ability ///
    + 0.6*age + 0.20*parent_edu + 0.15*family_income + u
gen ln_score = ln(score)

* ===== 第一部分：控制变量到底在控制什么 =====

* 任务1：遗漏变量比较（本段为讨论“控制”使用对 ai 线性的设定）
regress score ai age parent_edu family_income
estimates store omitted
local b_omit = _b[ai]
regress score ai age parent_edu family_income ability
estimates store full
display "遗漏ability时AI系数 = " `b_omit'
display "控制ability后AI系数 = " _b[ai]
estimates table omitted full, b(%9.4f) se(%9.4f) stats(N r2)

* 任务2：FWL —— 净 variation 解释净 variation
* 第1步：净化 ai
regress ai age parent_edu family_income ability
predict ai_resid, residuals
* 第2步：净化成绩
regress score age parent_edu family_income ability
predict score_resid, residuals
* 第3步：剩余解释剩余（不加常数项）
regress score_resid ai_resid, noconstant
display "FWL 第 3 步的AI系数 = " _b[ai_resid]
display "应与上面多元回归中的AI系数相同。"
local b_fwl = _b[ai_resid]
quietly summarize ai_resid
local xmin = r(min)
local xmax = r(max)
twoway (scatter score_resid ai_resid, mcolor(gs10)) ///
    (function y=`b_fwl'*x, range(`xmin' `xmax') lcolor(navy) lwidth(medthick)), ///
    title("净化后的 AI 使用时间对净化后的成绩") ///
    xtitle("净 AI 使用时间") ytitle("净成绩 variation") legend(off)
graph export output/ch05-fig2-fwl-net-variation.png, width(1800) replace

* ===== 第二部分：模型诊断 =====

* 任务1：多重共线性（在完整模型上计算）
estimates restore full
estat vif
corr parent_edu family_income

* 任务2：函数形式与异方差
estimates restore omitted
predict fitted, xb
predict resid, residuals
rvfplot, yline(0) title("成绩模型残差图：误差波动并不恒定")
graph export output/ch05-fig4-residual-heteroskedasticity.png, width(1800) replace
estat hettest
estat imtest, white
estat ovtest

* 任务3：稳健推断
regress score ai age parent_edu family_income, vce(robust)
estimates store robust
estimates table omitted robust, b(%9.4f) se(%9.4f) stats(N r2)

* 任务4：RESET拒绝线性形式后，恢复第4章的二次项与交互项
regress ln_score c.ai##c.ai i.strong c.ai#i.strong age parent_edu family_income ability, vce(robust)
estimates store nonlinear
margins strong, dydx(ai) at(ai=(5 10 15 20))
margins strong, dydx(ai)

display "诊断结论：稳健标准误修正异方差下的推断，但不会修复遗漏变量偏误。"
