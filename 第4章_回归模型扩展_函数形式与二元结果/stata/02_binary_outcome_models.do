* 第4章精简二元结果案例：LPM、Logit与Probit
* 沿用第4、5章的DGP，事先以75分为达标线。
* Stata与Python随机数发生器不同，小数无需逐一相同；两边报告同一估计对象。

clear all
set more off
set seed 20260915
capture mkdir output
set obs 800

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
gen pass = score >= 75
label variable pass "课程成绩是否达到75分"

summarize pass

* LPM：AI系数本身就是概率点变化，使用稳健标准误。
regress pass ai age i.strong, vce(robust)
estimates store lpm
predict p_lpm, xb
summarize p_lpm
count if p_lpm < 0
count if p_lpm > 1

* Logit：原始系数不是概率变化，报告AI的导数型AME。
logit pass ai age i.strong, vce(robust)
estimates store logit
margins, dydx(ai)
predict p_logit, pr
summarize p_logit

* Probit：同样报告AI的导数型AME。
probit pass ai age i.strong, vce(robust)
estimates store probit
margins, dydx(ai)
predict p_probit, pr
summarize p_probit

estimates table lpm logit probit, b(%9.4f) se(%9.4f) stats(N)
