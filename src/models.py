# src/models.py 封装机器学习模型的训练、评估和保存逻辑

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib # 用于保存和加载模型
from imblearn.over_sampling import SMOTE # 处理类别不平衡
from src.config import TARGET_COLUMN, RANDOM_STATE, MODEL_PATH
from src.data_loader import load_data # 用于测试加载数据

def train_model(X: pd.DataFrame, y: pd.Series, model_name: str = 'LogisticRegression', use_smote: bool = False, **kwargs):
    """
    训练机器学习模型。

    Args:
        X (pd.DataFrame): 特征数据集。
        y (pd.Series): 目标变量。
        model_name (str): 要训练的模型名称 ('LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'SVM')。
        use_smote (bool): 是否使用 SMOTE 处理类别不平衡。
        **kwargs: 传递给模型构造函数的额外参数。

    Returns:
        tuple: (训练好的模型, 训练集上的预测, 测试集上的预测, 测试集真实标签, 测试集特征)
    """
    print(f"--- 训练模型: {model_name} (SMOTE: {use_smote}) ---")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 处理类别不平衡 (SMOTE)
    if use_smote:
        print("正在使用 SMOTE 进行过采样...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"原始训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"SMOTE后训练集形状: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    model = None
    if model_name == 'LogisticRegression':
        model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', **kwargs) # 'liblinear' 适合小数据集和二元分类
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **kwargs)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=RANDOM_STATE, **kwargs)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=RANDOM_STATE, **kwargs)
    elif model_name == 'SVM':
        model = SVC(random_state=RANDOM_STATE, probability=True, **kwargs) # probability=True for ROC AUC
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

    model.fit(X_train, y_train)
    print(f"{model_name} 模型训练完成。")

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return model, y_train_pred, y_test_pred, y_test, X_test

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, y_train_pred: np.ndarray, y_test_pred: np.ndarray, model_name: str = "Model"):
    """
    评估机器学习模型并打印性能指标。

    Args:
        model: 训练好的模型。
        X_test (pd.DataFrame): 测试集特征。
        y_test (pd.Series): 测试集真实标签。
        y_train_pred (np.ndarray): 训练集预测标签。
        y_test_pred (np.ndarray): 测试集预测标签。
        model_name (str): 模型名称。

    Returns:
        dict: 包含模型评估指标的字典。
    """
    print(f"\n--- 模型评估: {model_name} ---")

    metrics = {}

    # 训练集性能
    train_accuracy = accuracy_score(y_test, y_test_pred) # 修正为用测试集
    train_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_test, y_test_pred)

    print("\n测试集性能指标:")
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test, y_test_pred):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
    
    # ROC AUC Score (需要模型支持 predict_proba)
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        metrics['ROC_AUC'] = roc_auc
    else:
        print("模型不支持 predict_proba，无法计算 ROC AUC。")

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_test_pred))

    metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['precision'] = precision_score(y_test, y_test_pred)
    metrics['recall'] = recall_score(y_test, y_test_pred)
    metrics['f1_score'] = f1_score(y_test, y_test_pred)
    metrics['confusion_matrix'] = cm

    return metrics

def save_model(model, path=MODEL_PATH):
    """
    保存训练好的模型。

    Args:
        model: 要保存的机器学习模型。
        path (str): 模型保存路径。
    """
    joblib.dump(model, path)
    print(f"模型已保存到 '{path}'。")

def load_model(path=MODEL_PATH):
    """
    加载保存的机器学习模型。

    Args:
        path (str): 模型文件路径。

    Returns:
        obj: 加载的机器学习模型。
    """
    try:
        model = joblib.load(path)
        print(f"模型已从 '{path}' 加载。")
        return model
    except FileNotFoundError:
        print(f"错误：模型文件未找到。请确保模型位于 '{path}'。")
        return None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None

if __name__ == "__main__":
    # 简单测试模型训练和评估功能
    from src.preprocessor import preprocess_data # 需要预处理器来获取数据

    print("--- 正在加载和预处理数据进行测试 ---")
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    if processed_df is not None:
        X = processed_df.drop(TARGET_COLUMN, axis=1)
        y = processed_df[TARGET_COLUMN]

        # 训练并评估逻辑回归模型
        print("\n--- 测试 Logistic Regression 模型 (无 SMOTE) ---")
        lr_model, lr_train_pred, lr_test_pred, y_test_lr, X_test_lr = train_model(X, y, model_name='LogisticRegression')
        evaluate_model(lr_model, X_test_lr, y_test_lr, lr_train_pred, lr_test_pred, model_name='LogisticRegression')
        save_model(lr_model, MODEL_PATH.replace('.pkl', '_lr.pkl'))

        # 训练并评估随机森林模型 (带 SMOTE)
        print("\n--- 测试 Random Forest 模型 (带 SMOTE) ---")
        rf_model, rf_train_pred, rf_test_pred, y_test_rf, X_test_rf = train_model(X, y, model_name='RandomForest', use_smote=True, n_estimators=100)
        evaluate_model(rf_model, X_test_rf, y_test_rf, rf_train_pred, rf_test_pred, model_name='RandomForest')
        save_model(rf_model, MODEL_PATH.replace('.pkl', '_rf_smote.pkl'))
    else:
        print("数据预处理失败，无法测试模型。")