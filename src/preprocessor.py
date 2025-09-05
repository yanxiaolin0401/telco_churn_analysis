# src/preprocessor.py
import os
import sys
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到Python路径
sys.path.append(project_root)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import TARGET_COLUMN, PROCESSED_DATA_PATH, RANDOM_STATE
from src.data_loader import load_data # 用于测试加载原始数据

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对电信客户流失数据集进行清洗和预处理。

    包括：
    1. 处理 'TotalCharges' 列的缺失值（空字符串）并转换为数值类型。
    2. 删除 'customerID' 列。
    3. 对二元分类特征进行标签编码 (Yes/No, Female/Male)。
    4. 对多元分类特征进行独热编码。
    5. 对数值特征进行标准化。

    Args:
        df (pd.DataFrame): 原始数据集。

    Returns:
        pd.DataFrame: 经过清洗和预处理后的数据集。
    """
    if df is None:
        print("输入DataFrame为空，无法进行预处理。")
        return None

    df_processed = df.copy()

    print("--- 开始数据预处理 ---")

    # 1. 处理 'TotalCharges' 列的缺失值（空字符串）并转换为数值类型
    # 替换空字符串为 NaN
    df_processed['TotalCharges'] = df_processed['TotalCharges'].replace(' ', np.nan)
    # 转换为数值类型（float），无法转换的会变成 NaN
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'])

    # 处理 TotalCharges 中的 NaN 值
    # 经探索，这11个NaN值对应的tenure（客户服务时长）都是0，说明是新客户，TotalCharges理应为0
    # 检查tenure是否为0
    nan_total_charges_indices = df_processed[df_processed['TotalCharges'].isnull()].index
    if not nan_total_charges_indices.empty:
        if all(df_processed.loc[nan_total_charges_indices, 'tenure'] == 0):
            df_processed['TotalCharges'].fillna(0, inplace=True)
            print(f"TotalCharges 列中的 {len(nan_total_charges_indices)} 个缺失值已根据 tenure=0 填充为 0。")
        else:
            # 如果有tenure不为0的，则填充中位数（以防万一）
            median_total_charges = df_processed['TotalCharges'].median()
            df_processed['TotalCharges'].fillna(median_total_charges, inplace=True)
            print(f"TotalCharges 列中的 {len(nan_total_charges_indices)} 个缺失值已填充为中位数 {median_total_charges}。")
    else:
        print("TotalCharges 列没有缺失值需要处理。")

    # 2. 删除 'customerID' 列——客户的唯一标识符，对预测流失没有直接价值
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
        print("已删除 'customerID' 列。")

    # 3. 对二元分类特征进行标签编码 (Yes/No, Female/Male)
    binary_features_map = {
        'gender': {'Female': 0, 'Male': 1}, # 或者 0/1 任意指定
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        TARGET_COLUMN: {'No': 0, 'Yes': 1} # 目标变量也进行编码
    }

    for col, mapping in binary_features_map.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
            print(f"已对二元特征 '{col}' 进行标签编码。")

    # 4. 对多元分类特征进行独热编码
    # 识别出需要独热编码的列
    # 首先获取所有object类型，排除已处理的二元特征和目标变量
    categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
    
    # 'MultipleLines' 和 InternetService 相关的特征中包含 'No phone service' 或 'No internet service'，
    # 这些可以看作是'No'的一种特殊情况，但独热编码会将其视为独立类别。
    # 如果要简化，可以先统一替换，但通常独热编码更直接。
    # 这里直接进行独热编码。

    print(f"即将对以下多元分类特征进行独热编码: {categorical_cols}")
    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
        #drop_first=False 意味着不会删除第一个哑变量，保留所有类别。这在某些情况下更有利于解释，但也会引入多重共线性（对于线性模型）。对于大多数树模型和PCA等，影响不大。
        print("已对多元分类特征进行独热编码。")
    else:
        print("没有需要进行独热编码的多元分类特征。")
        
    # 5. 对数值特征进行标准化
    # 排除目标变量和已编码的二元特征
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN) # 目标变量不参与标准化
    
    # 排除已是二元编码的 SeniorCitizen，因为它虽然是int64，但代表的是0/1的分类
    if 'SeniorCitizen' in numerical_cols:
         numerical_cols.remove('SeniorCitizen')
#使用 StandardScaler 将它们转换为均值为0，方差为1的分布。这对于许多机器学习算法（如逻辑回归、SVM、神经网络、K-Means等）至关重要，因为它可以防止具有较大尺度的特征在模型训练中占据主导地位。
    print(f"即将对以下数值特征进行标准化: {numerical_cols}")
    if numerical_cols:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        print("已对数值特征进行标准化。")
    else:
        print("没有需要进行标准化的数值特征。")

    print("--- 数据预处理完成 ---")
    return df_processed

if __name__ == "__main__":
    # 加载原始数据
    raw_df = load_data()

    if raw_df is not None:
        # 进行预处理
        processed_df = preprocess_data(raw_df)

        if processed_df is not None:
            print("\n预处理后的数据集前5行:")
            print(processed_df.head())
            print("\n预处理后的数据集信息:")
            processed_df.info()

            # 保存预处理后的数据
            processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"\n预处理后的数据已保存到 '{PROCESSED_DATA_PATH}'")
    else:
        print("原始数据加载失败，无法进行预处理。")