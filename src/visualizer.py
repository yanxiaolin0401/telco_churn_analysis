# src/visualizer.py
'''
  优化说明：
    将之前的绘图逻辑封装成独立的函数，每个函数负责绘制一种特定类型的图表。
    每个绘图函数都接受 save_path 参数，允许用户指定是否保存图片以及保存路径。
    添加了 plot_confusion_matrix 函数，以便可视化模型性能。
    修复了 seaborn.barplot 的 FutureWarning，通过明确指定 hue 参数。
    增加了中文支持的字体配置，并且在X轴标签处对常见二元特征提供了更具可读性的标签。
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from src.config import FIGURES_DIR, TARGET_COLUMN

# 设置图表风格和中文支持
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout() # 自动调整布局，防止标签重叠

# 确保 FIGURES_DIR 存在
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_churn_distribution(df: pd.DataFrame, target_column: str = TARGET_COLUMN, save_path: str = None):
    """
    绘制目标变量（流失）的分布图。

    Args:
        df (pd.DataFrame): 包含目标变量的数据集。
        target_column (str): 目标变量的列名。
        save_path (str, optional): 图片保存路径。如果为None，则不保存。
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df, palette='viridis')
    plt.title(f'{target_column} 分布', fontsize=14)
    plt.xlabel('是否流失 (0=否, 1=是)', fontsize=12)
    plt.ylabel('客户数量', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['不流失', '流失'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"流失分布图已保存到: {save_path}")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str = '特征相关性矩阵', save_path: str = None):
    """
    绘制特征相关性矩阵热力图。

    Args:
        df (pd.DataFrame): 包含特征的数据集。
        title (str): 图表标题。
        save_path (str, optional): 图片保存路径。
    """
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title, fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"相关性矩阵图已保存到: {save_path}")
    plt.show()

def plot_numerical_distributions(df: pd.DataFrame, features: list, hue_column: str = None, title_suffix: str = '', save_path: str = None):
    """
    绘制数值特征的分布图 (直方图和KDE)。

    Args:
        df (pd.DataFrame): 包含数值特征的数据集。
        features (list): 要绘制的数值特征列表。
        hue_column (str, optional): 用于区分的分类列名 (例如 'Churn')。
        title_suffix (str): 标题后缀。
        save_path (str, optional): 图片保存路径。
    """
    plt.figure(figsize=(5 * len(features), 5))
    for i, feature in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        if hue_column:
            sns.histplot(data=df, x=feature, hue=hue_column, kde=True, bins=30, palette='viridis', stat='density', common_norm=False)
            plt.title(f'{feature} 分布 ({hue_column} 区分){title_suffix}', fontsize=14)
            plt.xlabel(f'{feature} (标准化)', fontsize=12)
        else:
            sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
            plt.title(f'{feature} 分布{title_suffix}', fontsize=14)
            plt.xlabel(feature, fontsize=12)
        plt.ylabel('密度', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"数值特征分布图已保存到: {save_path}")
    plt.show()

def plot_categorical_churn_rates(df: pd.DataFrame, categorical_cols: list, target_column: str = TARGET_COLUMN, save_path: str = None):
    """
    绘制分类特征与流失率的关系条形图。

    Args:
        df (pd.DataFrame): 包含分类特征和目标变量的数据集。
        categorical_cols (list): 要绘制的分类特征列表 (应为0/1编码的列)。
        target_column (str): 目标变量的列名。
        save_path (str, optional): 图片保存路径。
    """
    num_cols = len(categorical_cols)
    num_rows = (num_cols + 2) // 3 # 每行3个图
    plt.figure(figsize=(18, 5 * num_rows))

    for i, col in enumerate(categorical_cols):
        plt.subplot(num_rows, 3, i + 1)
        # 确保独热编码列转换为int类型，因为groupby可能会对bool类型有不同的处理
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        churn_rate = df.groupby(col)[target_column].mean().reset_index()
        # 修复FutureWarning: 明确指定hue
        sns.barplot(x=col, y=target_column, data=churn_rate, palette='plasma', hue=col, legend=False)
        plt.title(f'{col} 对流失率的影响', fontsize=14)
        plt.ylabel('流失率', fontsize=12)
        plt.xlabel('')
        # 根据特征名提供更具描述性的X轴标签
        if col == 'gender':
            plt.xticks(ticks=[0, 1], labels=['女性', '男性'])
        elif col == 'SeniorCitizen':
            plt.xticks(ticks=[0, 1], labels=['非老年', '老年'])
        elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            plt.xticks(ticks=[0, 1], labels=['否', '是'])
        elif '_No internet service' in col:
            plt.xticks(ticks=[0, 1], labels=['非无互联网服务', '无互联网服务'])
        elif '_No phone service' in col:
            plt.xticks(ticks=[0, 1], labels=['非无电话服务', '无电话服务'])
        elif '_No' in col or '_Yes' in col:
            plt.xticks(ticks=[0, 1], labels=['否', '是']) # 独热编码的No/Yes
        else: # 对于其他独热编码的PaymentMethod, Contract等，直接用0/1
            plt.xticks(ticks=[0, 1], labels=['0', '1'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"分类特征流失率图已保存到: {save_path}")
    plt.show()


def plot_roc_curves(models_performance: dict, save_path: str = None):
    """
    绘制多个模型的ROC曲线。

    Args:
        models_performance (dict): 字典，每个键是模型名称，值是 (model, X_test, y_test) 的元组。
        save_path (str, optional): 图片保存路径。
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测 (AUC = 0.50)')

    for model_name, (model, X_test, y_test) in models_performance.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        else:
            print(f"模型 '{model_name}' 不支持 predict_proba，无法绘制ROC曲线。")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (假阳性率)', fontsize=12)
    plt.ylabel('True Positive Rate (真阳性率)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) 曲线', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲线图已保存到: {save_path}")
    plt.show()

def plot_feature_importance(feature_importances_df: pd.DataFrame, top_n: int = 15, title: str = '特征重要性', save_path: str = None):
    """
    绘制特征重要性条形图。

    Args:
        feature_importances_df (pd.DataFrame): 包含 'Feature' 和 'Importance' 列的DataFrame。
        top_n (int): 显示前N个最重要的特征。
        title (str): 图表标题。
        save_path (str, optional): 图片保存路径。
    """
    plt.figure(figsize=(12, 8))
    # 修复FutureWarning: 明确指定hue
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(top_n), palette='viridis', hue='Feature', legend=False)
    plt.title(f'Top {top_n} {title}', fontsize=16)
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"特征重要性图已保存到: {save_path}")
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = ['不流失', '流失'], title: str = '混淆矩阵', save_path: str = None):
    """
    绘制混淆矩阵。

    Args:
        y_true (np.ndarray): 真实标签。
        y_pred (np.ndarray): 预测标签。
        class_names (list): 类别名称列表。
        title (str): 图表标题。
        save_path (str, optional): 图片保存路径。
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵图已保存到: {save_path}")
    plt.show()