import os

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据文件路径
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'telco_customer_churn.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_churn_data.csv')

# 模型文件路径
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_prediction_model.pkl')

# 报告和图表路径
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# 目标变量名称
TARGET_COLUMN = 'Churn'

# 随机种子，用于复现性
RANDOM_STATE = 42

#随机种子(random seed)是用来初始化伪随机数生成器的起始值。
# 在机器学习算法、数据抽样、模型训练等过程中，我们经常需要生成随机数或进行随机操作。
# 通过固定随机种子，我们可以确保每次运行代码时生成的随机序列都是相同的，这样就能够复现之前的结果。

print(f"项目根目录: {BASE_DIR}")
print(f"原始数据路径: {RAW_DATA_PATH}")