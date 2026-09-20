* 第4章案例：AI 使用与课程成绩——回归模型扩展：函数形式与二元结果
* 本脚本生成第4、5章共用的一份教学数据并完成第4章的分析。
* 第5章的控制与诊断见 ch05/01_ai_score_diagnostics.do。
* Stata 与 Python 的随机数发生器不同，两边样本不会逐个相同，但数据特征与结论一致。

clear all
set more off
set seed 20260915
capture mkdir output
set obs 800

* —— 第4、5章共用的一份数据 ——
gen ability = rnormal()
gen age = min(max(round(rnormal(20,1.6)),17),26)
gen female = runiform() < 0.5
gen family_bg = rnormal()
gen parent_edu = 12 + 2.2*family_bg + rnormal(0,0.30)
gen family_income = 8 + 1.9*family_bg + rnormal(0,0.30)
gen ai = min(max(3 + 3.2*ability + 0.25*age - 1.2*female + rnormal(0,3.5),0),30)
gen ai2 = ai^2
gen ai_female = ai*female
gen sigma = 2.0 + 0.35*ai
gen u = rnormal(0,sigma)
gen score = 56 + 1.20*ai - 0.060*ai2 + 0.55*ai_female + 4.5*ability ///
    + 0.6*age - 1.8*female + 0.20*parent_edu + 0.15*family_income + u
gen ln_score = ln(score)

* 任务1：成绩水平模型与对数成绩模型
regress score ai age i.female
estimates store level
regress ln_score ai age i.female
estimates store log

* 任务2：AI 使用时间的二次项
regress ln_score ai c.ai##c.ai age i.female
estimates store quadratic
display "对数成绩模型中 AI 的极值点 = " -_b[ai]/(2*_b[c.ai#c.ai])
margins, at(ai=(0(2)24) age=(20) female=(0))
marginsplot, recast(line) recastci(rarea) ///
    title("AI 使用时间与预测对数成绩") xtitle("每周 AI 使用时间（小时）") ytitle("预测对数成绩")
graph export output/ch04-fig7-ai-score-curve.png, width(1800) replace

* 任务3：AI 使用时间与性别的交互项
regress ln_score c.ai##i.female c.ai##c.ai age
estimates store interaction
margins female, dydx(ai)
margins female, dydx(ai) at(ai=(5 10 15 20))
margins female, at(ai=(0(2)24) age=(20))
marginsplot, recast(line) recastci(rarea) ///
    title("AI 使用与成绩：不同性别的两条预测曲线") ///
    xtitle("每周 AI 使用时间（小时）") ytitle("预测对数成绩")
graph export output/ch04-fig8-ai-gender.png, width(1800) replace

estimates table level log quadratic interaction, b(%9.4f) se(%9.4f) stats(N r2)
