from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, fbeta_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def quantify_binary_classification(df, y_pred,label_col ,beta_list=[1, 0.5]):
    """
    ประเมินผล classification: precision, recall, f1, f-beta และ confusion matrix

    Parameters:
        df (pd.DataFrame): DataFrame ที่มีค่าจริงใน true_col
        y_pred (array-like): ค่าทำนายภายนอก (optional) ถ้าไม่ใส่ จะใช้ df['pred']
        true_col (str): ชื่อคอลัมน์ค่าจริงใน df
        beta_list (list): beta ต่าง ๆ สำหรับคำนวณ f-beta

    Returns:
        dict: ค่าประเมินทั้งหมด
    """
    y_true = df[label_col].values
    y_pred = df[y_pred].values

    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    for b in beta_list:
        results[f"f{b}_score"] = fbeta_score(y_true, y_pred, beta=b, zero_division=0)

    return results
