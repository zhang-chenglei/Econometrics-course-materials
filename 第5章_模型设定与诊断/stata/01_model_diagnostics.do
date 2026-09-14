* 第5章案例：一份工资回归的完整诊断
clear all
set more off
set seed 20260912
capture mkdir output
set obs 800

gen ability = rnormal()
gen educ = min(max(14 + 0.8*ability + rnormal(0,1.4),9),22)
gen exper = min(max(rnormal(14,6),0),35)
gen female = runiform() < 0.5
gen city_factor = rnormal()
gen city_income = city_factor + rnormal(0,0.15)
gen city_size = city_factor + rnormal(0,0.15)
gen sigma = 0.18 + 0.018*exper
gen u = rnormal(0,sigma)
gen ln_wage = 2 + 0.08*educ + 0.035*exper - 0.10*female ///
    + 0.10*ability + 0.05*city_income + 0.04*city_size + u

* 任务1：遗漏变量比较
regress ln_wage educ exper female city_income city_size
estimates store omitted
local b_omit = _b[educ]
regress ln_wage educ exper female city_income city_size ability
estimates store full
display "遗漏ability时教育系数 = " `b_omit'
display "控制ability后教育系数 = " _b[educ]
estimates table omitted full, b(%9.4f) se(%9.4f) stats(N r2)

* 任务2：多重共线性
estat vif
corr city_income city_size

* 任务3：异方差、稳健推断与最终诊断
estimates restore omitted
predict fitted, xb
predict resid, residuals
rvfplot, yline(0) title("工资模型残差图：误差波动并不恒定")
graph export output/ch05-residual-heteroskedasticity.png, width(1800) replace
estat hettest
estat imtest, white

regress ln_wage educ exper female city_income city_size, vce(robust)
estimates store robust
estimates table omitted robust, b(%9.4f) se(%9.4f) stats(N r2)

display "诊断结论：稳健标准误修正异方差下的推断，但不会修复遗漏变量偏误。"

