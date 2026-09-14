"""生成第11—14章的可执行教学笔记本。"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    11: {
        "title": "第11章：从政策事实到研究设计",
        "goal": "核对政策批次、样本覆盖和识别威胁，把研究问题变成可执行的设计书。",
        "script": "ch11_research_design.py",
        "tables": ["表11-1_政策事实表.csv", "表11-4_研究设计验收.csv", "表11-5_识别威胁与证据地图.csv"],
        "figures": ["图11-1_试验区批复时间线.png", "图11-2_各试点企业覆盖.png"],
    },
    12: {
        "title": "第12章：数据准备与变量构建",
        "goal": "检查唯一性、缺失、分布、处理支持和平衡性，生成可进入估计环节的分析数据诊断。",
        "script": "ch12_data_preparation.py",
        "tables": ["表12-1_样本筛选流程.csv", "表12-2_描述性统计.csv", "表12-4_政策前特征平衡.csv"],
        "figures": ["图12-1_主要结果变量分布.png", "图12-2_处理组与对照组原始趋势.png", "图12-3_政策前特征平衡图.png"],
    },
    13: {
        "title": "第13章：模型估计与结果检验",
        "goal": "依次完成TWFE教学基准、交错DID、动态效应、稳健性、异质性和机制相关检验。",
        "script": "ch13_estimation.py",
        "tables": ["表13-1_TWFE教学基准.csv", "表13-2_交错DID总体ATT.csv", "表13-5_平行趋势联合检验.csv", "表13-10_机制分析.csv", "表13-10b_机制交错DID.csv"],
        "figures": ["图13-1_动态事件研究.png", "图13-3_稳健性系数图.png", "图13-5_异质性系数图.png", "图13-6_机制结果.png"],
    },
    14: {
        "title": "第14章：研究成果的规范呈现",
        "goal": "把前章输出整理为统一数字口径的论文式表图和一份克制、可追溯的教学研究报告。",
        "script": "ch14_reporting.py",
        "tables": ["表14-1_TWFE教学基准最终版.csv", "表14-2_交错DID主结果最终版.csv", "表14-6_全文关键数字一致性检查.csv"],
        "figures": ["图14-1_主结果比较.png", "图14-2_事件研究最终版.png", "图14-3_稳健性结果摘要.png", "图14-4_机制证据摘要.png"],
    },
}


for chapter, config in CONFIG.items():
    chapter_dir = ROOT / f"第{chapter}章"
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}

    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# {config['title']}\n\n"
            f"> **材料性质**：真实政策背景与半合成企业教学样本。所有估计只用于教学，不用于评价真实政策。\n\n"
            "## Goal\n\n"
            f"{config['goal']}"
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "下面固定随机环境和绘图后端，并定位本章脚本及输出目录。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, runpy\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n\n"
            "start = Path.cwd().resolve()\n"
            "PACKAGE = next((p for p in [start, *start.parents] "
            "if (p / '共享代码').is_dir() and (p / '第11章').is_dir()), None)\n"
            "if PACKAGE is None:\n"
            "    raise FileNotFoundError('请从 comprehensive_case 目录或其子目录启动 Jupyter')\n"
            f"CHAPTER = PACKAGE / '第{chapter}章'\n"
            "os.environ['MPLBACKEND'] = 'Agg'\n"
            "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
            "os.environ['OMP_NUM_THREADS'] = '1'\n"
            "print(CHAPTER)"
        ),
        nbf.v4.new_markdown_cell(
            "## Steps\n\n"
            "运行本章完整脚本。脚本会覆盖本章表格、图形和说明文件，确保笔记本展示与文件夹中的交付物来自同一次计算。"
        ),
        nbf.v4.new_code_cell(
            f"script = CHAPTER / '代码' / '{config['script']}'\n"
            "runpy.run_path(str(script), run_name='__main__')\n"
            "print('本章脚本运行完成。')"
        ),
        nbf.v4.new_markdown_cell(
            "## Checks\n\n"
            "先检查关键结果表。完整机器可读结果仍以本章“表格”目录中的CSV为准。"
        ),
    ]

    for table in config["tables"]:
        notebook["cells"].append(
            nbf.v4.new_code_cell(
                f"display(Markdown('### {table.removesuffix('.csv')}'))\n"
                f"table = pd.read_csv(CHAPTER / '表格' / '{table}', encoding='utf-8-sig')\n"
                "display(table)"
            )
        )

    notebook["cells"].append(
        nbf.v4.new_markdown_cell(
            "下面核对本章主要图形。图中的标题、坐标、置信区间和图注应能脱离正文独立理解。"
        )
    )
    for figure in config["figures"]:
        notebook["cells"].append(
            nbf.v4.new_code_cell(
                f"display(Markdown('### {figure.removesuffix('.png')}'))\n"
                f"display(Image(filename=str(CHAPTER / '图形' / '{figure}'), width=900))"
            )
        )

    notebook["cells"].append(
        nbf.v4.new_markdown_cell(
            "## Next Steps\n\n"
            "1. 将结果与本章正文中的处理定义、样本口径和解释边界逐项核对。\n"
            "2. 复核所有不显著或相互冲突的证据，不删除不符合预期的规格。\n"
            "3. 若替换数据版本，按第11章到第14章的顺序重新运行全部笔记本。"
        )
    )

    target = chapter_dir / "代码" / f"第{chapter}章_可执行复现.ipynb"
    nbf.write(notebook, target)
    print(target)
