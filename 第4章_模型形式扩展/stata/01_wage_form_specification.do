* 第4章案例：从直线到曲线——工资模型的形式扩展
clear all
set more off
set seed 20260911
capture mkdir output
set obs 800

gen female = runiform() < 0.5
gen educ = min(max(rnormal(14,2),9),22)
gen exper = min(max(rnormal(15,7),0),38)
gen u = rnormal(0,0.28)
gen ln_wage = 5 + 0.075*educ + 0.050*exper - 0.0009*exper^2 ///
       - 0.12*female + 0.018*educ*female + u
gen wage = exp(ln_wage)

* 任务1：水平模型与对数模型
regress wage educ exper i.female
estimates store level
regress ln_wage educ exper i.female
estimates store log

* 任务2：经验平方项
regress ln_wage educ c.exper##c.exper i.female
estimates store quadratic
display "经验曲线极值点 = " -_b[exper]/(2*_b[c.exper#c.exper])
margins, at(exper=(0(2)38) educ=(14) female=(0))
marginsplot, recast(line) recastci(rarea) ///
    title("工作经验与预测对数工资") xtitle("工作经验（年）") ytitle("预测对数工资")
graph export output/ch04-experience-curve.png, width(1800) replace

* 任务3：教育与性别交互项
regress ln_wage c.educ##i.female c.exper##c.exper
estimates store interaction
margins female, dydx(educ)
margins female, at(educ=(9(1)22) exper=(15))
marginsplot, recast(line) recastci(rarea) ///
    title("教育回报的不同：一个交互项的直观表达") ///
    xtitle("教育年限") ytitle("预测对数工资")
graph export output/ch04-education-gender.png, width(1800) replace

estimates table level log quadratic interaction, b(%9.4f) se(%9.4f) stats(N r2)
