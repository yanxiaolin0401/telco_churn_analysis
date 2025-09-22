# 电信客户流失数据分析及可视化

本项目旨在对电信客户流失数据进行深入分析，包括数据清洗、探索性数据分析、机器学习建模（流失预测与聚类）以及结果可视化。电信客户流失：https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 项目结构

- `data/`: 存放原始数据 raw 和处理后的数据 processed
- `notebooks/`: Jupyter Notebooks，用于实验和探索
- `src/`: 核心 Python 代码模块
- `models/`: 存放训练好的模型
- `reports/`: 存放生成的报告和图表

## 设置与运行

1. 克隆仓库。
2. 创建并激活 Python 虚拟环境。

   创建虚拟环境

   `python -m venv .venv`

   激活虚拟环境 (Windows PowerShell)

   `.venv\Scripts\Activate.ps1`

   激活虚拟环境 (Windows Command Prompt / Git Bash)

   `.venv\Scripts\activate`

3. 安装依赖：`pip install -r requirements.txt`
4. 下载`telco_customer_churn.csv`数据集并放置到`data/raw/`目录下。并运行 preprocessor.py 脚本清洗数据
5. 运行 Jupyter Notebooks 1，2，3，4。
