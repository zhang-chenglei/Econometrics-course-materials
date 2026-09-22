version 17.0
clear all
set more off

* 第4讲 Agent Lab：企业使用 AI 能否提高创新产出？
* 先从课程页面下载 firm_ai_innovation_case.dta，
* 再将 Stata 工作目录切换到数据所在文件夹后运行。

use "firm_ai_innovation_case.dta", clear

* Step 1：认识数据
describe
summarize innovation_score ai_intensity prior_innovation rd_intensity ///
    digital_capability ln_employment firm_age export_share

* Step 2：先画图，再回归
histogram ai_intensity, frequency title("企业 AI 使用强度")
twoway (scatter innovation_score ai_intensity, mcolor(%35)) ///
       (lfit innovation_score ai_intensity), ///
       title("企业 AI 使用与创新产出")

* Step 3：简单回归——这里只能解释为样本中的相关关系
regress innovation_score ai_intensity, vce(robust)
estimates store naive

* Think First：运行下一行以前，先预测 ai_intensity 系数会怎样变化。
* 在这里写下你的预测：
* ________________________________________________________________

* Step 4：加入常见企业特征，观察相关系怎样变化
regress innovation_score ai_intensity prior_innovation rd_intensity ///
    digital_capability ln_employment firm_age export_share i.industry, ///
    vce(robust)
estimates store adjusted

* Step 5：并排比较。加入控制变量不自动构成因果识别。
estimates table naive adjusted, ///
    b(%9.3f) se(%9.3f) stats(N r2)

* 请另存一份自己的 do-file，在其中补充：
* 1. 每一步的运行结果；
* 2. 对系数含义和识别边界的解释；
* 3. 三种研究设计的比较；
* 4. Research Design Card 中提出的新研究设计。
