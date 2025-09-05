# src/data_loader.py加载原始数据。

import pandas as pd
from src.config import RAW_DATA_PATH

def load_data(path=RAW_DATA_PATH):
    """
    加载电信客户流失数据集。

    Args:
        path (str): 数据文件的路径。默认为 config.RAW_DATA_PATH。

    Returns:
        pd.DataFrame: 加载的数据集。
    """
    try:
        df = pd.read_csv(path)
        print(f"数据已成功从 '{path}' 加载。")
        return df
    except FileNotFoundError:
        print(f"错误：文件未找到。请确保数据集位于 '{path}'。")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

if __name__ == "__main__":
    # 简单测试数据加载功能
    df = load_data()
    if df is not None:
        print("\n数据集前5行:")
        print(df.head())
        print("\n数据集信息:")
        df.info()