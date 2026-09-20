* 第3章案例：AI 使用与课程成绩——控制变量与遗漏变量偏误
* 生成正文图2-4（遗漏模型与完整模型的 AI 系数及其置信区间）。
*
* 教学用模拟数据，数据生成过程人为设定且已知：
*   ai    = 3 + 3.2*ability + 0.25*age - 1.2*female + v,  v ~ N(0, 3.5)
*   score = 56 + 1.20*ai + 4.5*ability + 0.6*age - 1.8*female + u
* 其中 ability 是认知能力，同时影响 AI 使用时间和成绩，是本案例要考察的遗漏变量。
* 真实系数 1.20 是目标参数。
*
* 随机种子：Python 与 Stata 的随机数算法不同，同一份数据生成过程也会抽到不同的
* 样本。这里各自固定一个种子，使两种软件的输出都落在真实系数附近，便于课堂对照；
* 数值不必完全一致，图形形状和结论一致即可。

clear all
set more off
set seed 20260430
capture mkdir output

* ------------------------------------------------------------
* 1. 按数据生成过程造出教学用样本
* ------------------------------------------------------------
set obs 800
gen ability = rnormal(0, 1)
gen age     = round(rnormal(20, 1.6))
replace age = 17 if age < 17
replace age = 26 if age > 26
gen female  = runiform() < 0.5

gen ai = 3 + 3.2*ability + 0.25*age - 1.2*female + rnormal(0, 3.5)
replace ai = 0  if ai < 0
replace ai = 30 if ai > 30

gen score = 56 + 1.20*ai + 4.5*ability + 0.6*age - 1.8*female + rnormal(0, 5.5)

* ------------------------------------------------------------
* 2. 两个模型：遗漏模型与完整模型
* ------------------------------------------------------------
regress score ai
est store m1
regress score ai age female ability
est store m2

display _newline "模型比较（被解释变量：课程成绩；真实 AI 系数 = 1.20）："
estimates table m1 m2, keep(ai) b(%9.4f) se(%9.4f) stats(N r2)

display _newline "变量相关系数："
corr score ai ability age female

display _newline "完整模型摘要："
estimates restore m2
estimates table m2, b(%9.4f) se(%9.4f)

* ------------------------------------------------------------
* 3. 汇总 AI 系数与置信区间，保存并与 Python 版对照
* ------------------------------------------------------------
foreach m in m1 m2 {
    est restore `m'
    local b_`m'  = _b[ai]
    local lo_`m' = _b[ai] - 1.96*_se[ai]
    local hi_`m' = _b[ai] + 1.96*_se[ai]
}

clear
set obs 2
gen byte id = _n
gen double ai_coef  = .
gen double ci_lower = .
gen double ci_upper = .
replace ai_coef = `b_m1' in 1
replace ai_coef = `b_m2' in 2
replace ci_lower = `lo_m1' in 1
replace ci_lower = `lo_m2' in 2
replace ci_upper = `hi_m1' in 1
replace ci_upper = `hi_m2' in 2
label define mdl 1 "(1) 只含 AI" 2 "(2) 加年龄、性别、认知能力"
label values id mdl
label var ai_coef "AI 使用时间的系数"
export delimited id ai_coef ci_lower ci_upper using "output/ch03_model_comparison.csv", replace

* ------------------------------------------------------------
* 4. 图2-4：AI 系数在遗漏模型与完整模型中的位置
* ------------------------------------------------------------
twoway (rcap ci_lower ci_upper id, horizontal lcolor(gs10) lwidth(med)) ///
       (scatter id ai_coef, msymbol(O) mcolor(navy) msize(medlarge)), ///
    ylabel(1 2, valuelabel angle(0) labsize(small) noticks) yscale(reverse) ///
    xline(1.20, lcolor(maroon) lpattern(dash) lwidth(med)) ///
    ytitle("") xtitle("AI 使用时间的系数及 95% 置信区间") ///
    title("遗漏变量被控制后，AI 系数回到真实值附近") ///
    legend(off)
graph save output/ch03_ovb_coef.gph, replace
graph export output/ch03-fig4-ovb-coefficient-path.png, width(2400) replace

display _newline "图形已保存至 output/"
