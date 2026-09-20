/*══════════════════════════════════════════════════════════════
  第9章 现代因果推断与准实验方法 — 案例实践（B部分：断点回归）
  配套教材：《计量经济学：理论与实践》
  功能：生成虚构退休资格与消费数据，依次完成两个RDD任务：
        任务3：资格规则对结果的简化式跳跃
        任务4：模糊断点回归与稳健性检验（Fuzzy RDD）
  ══════════════════════════════════════════════════════════════*/

clear
set more off
set seed 20250915
capture mkdir output

*============================================================
* 数据生成：虚构的60岁退休资格规则与消费
* 2000个体，年龄按连续值记录
*============================================================
set obs 2000

gen age = runiform()*15 + 55          // 年龄55-70
gen running = age - 60                // 运行变量（中心化）
gen D = (age >= 60)                   // 断点指示变量

* 模糊断点：退休概率随年龄平滑上升，并在60岁额外跳跃
* 不设置65岁的第二个断点，避免污染较宽带宽下的60岁估计
gen retired_prob = 0.1 + 0.01*running + 0.7*D
replace retired_prob = max(0.02, min(0.98, retired_prob))
gen retired = rbinomial(1, retired_prob)

* 控制变量
gen income     = rnormal(80000, 20000) - 3000*(age-60)  // 处理前收入
gen education  = rnormal(12, 3)
gen health     = max(10, min(100, 80 - 0.5*(age-55) + rnormal(0, 5)))
gen family_sz  = max(1, min(6, round(rnormal(3, 1))))
gen urban      = rbinomial(1, 0.6)

* 因变量：年消费额
* 退休导致消费下降约4000
gen consumption = 50000 + 500*(age-55) + 0.3*income + 200*education  ///
                  + 50*health - 500*family_sz + 2000*urban           ///
                  - 4000*retired + rnormal(0, 5000)

label variable age          "年龄"
label variable running      "运行变量（age-60）"
label variable D            "达到60岁=1"
label variable retired      "已退休=1"
label variable consumption  "年消费额（元）"
label variable income       "年收入（元）"
label variable education    "教育年限"
label variable health       "健康指数（0-100）"

sum age running D retired consumption

* ――――――――――――――――――――――――――――――――――――――――――――
* 任务3：资格规则对结果的简化式跳跃
* ――――――――――――――――――――――――――――――――――――――――――――
display _n "========== 任务3：资格规则对结果的简化式跳跃 =========="

* 3a：可视化
display _n "--- 消费-年龄散点+拟合 ---"
twoway (scatter consumption age, msize(tiny) color(gs12))          ///
       (lpoly consumption age if age<60, bw(1) lcolor(blue))       ///
       (lpoly consumption age if age>=60, bw(1) lcolor(red)),      ///
       xline(60, lpattern(dash) lcolor(black))                     ///
       legend(off) ytitle("年消费额") xtitle("年龄")               ///
       title("年龄与消费的关系：断点在60岁")
graph export output/ch09-fig2-rdd-scatter.png, width(2400) replace

* 3b：简化式RDD估计（不同带宽）
display _n "--- 资格指示D对结果的简化式跳跃 ---"
tempname fr
file open `fr' using "output/ch09_rdd_reduced_form.csv", write replace
file write `fr' "带宽,D系数(简化式),标准误,样本量" _n
foreach h in 3 5 7 {
    reg consumption running D c.running#c.D if abs(running) <= `h', robust
    display "  带宽 h=`h': D系数 = " %9.2f _b[D] " (se=" %9.2f _se[D] ")"
    file write `fr' "h=`h'," %9.4f (_b[D]) "," %9.4f (_se[D]) "," %9.0f (e(N)) _n
}
file close `fr'

* 3c：使用rdrobust（如果已安装）
capture rdrobust consumption running, c(0)
if _rc == 0 {
    display "  rdrobust 自动带宽+偏差修正结果"
}
else {
    display "  （rdrobust 未安装，使用手动带宽估计）"
}

display _n "=== 提示 ==="
display "  D系数是资格规则对结果的简化式跳跃"
display "  因退休并不完全由门槛决定，它不是退休本身的处理效应"

* ――――――――――――――――――――――――――――――――――――――――――――
* 任务4：模糊断点回归与稳健性检验（Fuzzy RDD）
* ――――――――――――――――――――――――――――――――――――――――――――
display _n "========== 任务4：Fuzzy RDD =========="

* 4a：第一阶段——验证断点影响退休概率
display _n "--- 第一阶段：退休概率在断点处的跳跃 ---"
reg retired running D c.running#c.D if abs(running) <= 5, robust
display "  D系数 = " %9.3f _b[D] "（断点处退休概率的跳跃）"

* 4b：Fuzzy RDD（2SLS）
display _n "--- Fuzzy RDD 估计（不同带宽）---"
* 第一阶段跳跃（h=5），单独记一份
reg retired running D c.running#c.D if abs(running) <= 5, robust
local fs_jump = _b[D]

tempname ff
file open `ff' using "output/ch09_rdd_fuzzy_late.csv", write replace
file write `ff' "带宽,LATE(退休),标准误" _n
foreach h in 3 5 7 {
    gen running_D = running*D
    ivregress 2sls consumption running running_D (retired = D) if abs(running) <= `h', robust
    display "  带宽 h=`h': retired系数 = " %9.2f _b[retired] " (se=" %9.2f _se[retired] ")"
    file write `ff' "h=`h'," %9.4f (_b[retired]) "," %9.4f (_se[retired]) _n
    drop running_D
}
file close `ff'

* 4c：加入控制变量
display _n "--- Fuzzy RDD + 控制变量 ---"
gen running_D = running*D
ivregress 2sls consumption running running_D income education health family_sz urban ///
    (retired = D) if abs(running) <= 5, robust
display "  retired系数（含控制变量）= " %9.2f _b[retired] " (se=" %9.2f _se[retired] ")"
local late_ctrl = _b[retired]

* 4d：安慰剂检验（假断点=58和62）
display _n "--- 安慰剂检验：假断点 ---"
gen running58 = age - 58
gen fake58 = (age >= 58)
reg consumption running58 fake58 c.running58#c.fake58 if age < 60 & abs(running58) <= 5, robust
local b58 = _b[fake58]
local p58 = 2*ttail(e(df_r),abs(_b[fake58]/_se[fake58]))
display "  假断点58岁: D系数 = " %9.2f `b58' " (p=" %6.4f `p58' ")"

gen running62 = age - 62
gen fake62 = (age >= 62)
reg consumption running62 fake62 c.running62#c.fake62 if age > 60 & abs(running62) <= 5, robust
local b62 = _b[fake62]
local p62 = 2*ttail(e(df_r),abs(_b[fake62]/_se[fake62]))
display "  假断点62岁: D系数 = " %9.2f `b62' " (p=" %6.4f `p62' ")"

* 4e：协变量平滑性检验
display _n "--- 协变量平滑性检验（断点处不应有跳跃）---"
tempname fc
file open `fc' using "output/ch09_rdd_covariate_smoothness.csv", write replace
file write `fc' "协变量,断点跳跃,p值" _n
foreach var of varlist income education health family_sz urban {
    reg `var' running D c.running#c.D if abs(running) <= 5, robust
    local p = 2*ttail(e(df_r), abs(_b[D]/_se[D]))
    display "  `var': p(D) = " %6.4f `p' "  " _continue
    if `p' < 0.05 display "⚠ 有跳跃！" _continue
    display ""
    file write `fc' ("`var'") "," %9.4f (_b[D]) "," %9.4f (`p') _n
}
file close `fc'

tempname fp
file open `fp' using "output/ch09_rdd_placebo.csv", write replace
file write `fp' "假断点,D系数,p值" _n
file write `fp' "58岁," %9.4f (`b58') "," %9.4f (`p58') _n
file write `fp' "62岁," %9.4f (`b62') "," %9.4f (`p62') _n
file close `fp'

display _n "结果已保存至 output/"

display _n "=== 提示 ==="
display "  资格指示D的结果跳跃 = 简化式资格效应（常称ITT）"
display "  Fuzzy RDD的retired系数 = LATE（局部平均处理效应）"
display "  Fuzzy RDD = 简化式跳跃 / 第一阶段概率跳跃"
display "  协变量若未见明显跳跃，与连续性相容；较大的p值不能证明个体必然可比"
display "  假断点若未见明显跳跃可作辅助证据，但不能单独证明政策效应"
display "  以上判断以你自己的实际输出为准：本次运行的逐项数值见 output/ 下的 CSV，"
display "  出现显著跳跃时，应把它当作对设计的质疑，而不是可以忽略的噪声"
display ""
display "  RDD标准流程："
display "  步骤1: 散点+拟合图 → 视觉确认断点跳跃"
display "  步骤2: 简化式RDD多带宽 → 资格效应"
display "  步骤3: 第一阶段 → 验证断点影响处理概率"
display "  步骤4: Fuzzy RDD → LATE估计"
display "  步骤5: 协变量平滑性 → 检查连续性假设是否受到明显质疑"
display "  步骤6: 假断点安慰剂 → 检查是否存在其他异常跳跃"
