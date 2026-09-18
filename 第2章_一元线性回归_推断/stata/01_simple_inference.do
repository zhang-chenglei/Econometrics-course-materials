* 第2章案例：一元回归推断与样本波动
clear all
set more off
set seed 20260914
set obs 800

gen ai = min(max(rnormal(7.5,3),0),18)
gen score = 70 + 1.2*ai + rnormal(0,5.5)

reg score ai
estimates store full
matrix T = r(table)'
matlist T

capture mkdir output
preserve
clear
svmat double T, names(col)
gen term = ""
replace term = "_cons" in 1
replace term = "ai" in 2
export delimited using "output/ch02_regression_table.csv", replace
restore

gen random_order = runiform()
sort random_order
gen half = _n > _N/2
reg score ai if half==0
reg score ai if half==1

* 系数区间图与更多重复切分可用配套Python脚本直接生成。
