version 17.0
clear all
set more off

* 第4讲 Agent Lab：AI 使用真的能提高学生成绩吗？
* 先从课程页面同时下载 AI_score_case.dta，将两个文件放在同一文件夹，
* 再把 Stata 工作目录切换到该文件夹后运行。

use "AI_score_case.dta", clear

* Step 1：认识数据
describe
summarize final_score ai_hours prior_score motivation digital_skill ///
    study_hours assignment_quality

* Step 2：先画图，再回归
histogram ai_hours, frequency title("每周 AI 使用小时数")
twoway (scatter final_score ai_hours, mcolor(%35)) ///
       (lfit final_score ai_hours), ///
       title("AI 使用与期末成绩")

* Step 3：天真回归——这里只能先写成条件相关
regress final_score ai_hours, vce(robust)
estimates store naive

* Think First：运行下一行以前，先预测 ai_hours 系数会向哪个方向变化。
* 在这里写下你的预测：
* ________________________________________________________________

* Step 4：加入常见控制变量，观察相关系数怎样变化
regress final_score ai_hours prior_score motivation digital_skill ///
    study_hours i.gender i.class_id, vce(robust)
estimates store adjusted

* Step 5：并排比较。控制变量改变相关系数，但不自动构成识别策略。
estimates table naive adjusted, ///
    b(%9.3f) se(%9.3f) stats(N r2)

* 请另存一份自己的 do-file，在其中补充：
* 1. 每一步的运行结果；
* 2. 对系数含义和识别边界的解释；
* 3. 三种研究设计的比较；
* 4. Research Design Card 中提出的新研究设计。
