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

print(f"项目根目录: {BASE_DIR}")
print(f"原始数据路径: {RAW_DATA_PATH}")