from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, fbeta_score

def isolation_forest(df, contamination=0.05, n_estimators=100, max_samples='auto', random_state=42):
    """
    รัน Isolation Forest บน DataFrame ที่เป็นข้อมูลเชิงตัวเลขล้วน

    Parameters:
    - df : pd.DataFrame — ข้อมูลที่เตรียมพร้อมแล้ว
    - contamination : float — สัดส่วนที่คาดว่าจะเป็น anomaly (default = 0.05)
    - n_estimators : int — จำนวน trees ใน forest (default = 100)
    - max_samples : int, float หรือ 'auto' — จำนวน sample ต่อ tree (default = 'auto')
    - random_state : int — สำหรับ reproducibility (default = 42)

    Returns:
    - result_df : pd.DataFrame — พร้อมคอลัมน์ anomaly (0/1) และ anomaly_score
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state
    )
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result_df = df.copy()
    result_df["anomaly"] = (preds == -1).astype(int)
    result_df["anomaly_score"] = -scores  # ยิ่งสูงยิ่งเสี่ยงผิดปกติ

    return result_df

def evaluate_if_performance_with_best(
    df_input,
    df_true,
    label_col='fraud_reported',
    contamination_list=[0.01, 0.03, 0.05, 0.1],
    n_estimators_list=[50, 100, 200],
    max_samples_list=['auto', 0.6, 0.8, 1.0],
    beta=0.5
):
    if df_true[label_col].dtype == object:
        y_true = df_true[label_col].map({'Y': 1, 'N': 0}).astype(int)
    else:
        y_true = df_true[label_col].astype(int)

    results = []

    for contamination in contamination_list:
        for n_estimators in n_estimators_list:
            for max_samples in max_samples_list:
                try:
                    result_df = isolation_forest(
                        df_input,
                        contamination=contamination,
                        n_estimators=n_estimators,
                        max_samples=max_samples
                    )
                    y_pred = result_df['anomaly']
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

                    results.append({
                        'model': f'Isolation Forest | contamination: {contamination}, n_estimators: {n_estimators}, max_samples: {max_samples}',
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        f'f{beta}': fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                        'anomaly_count': y_pred.sum()
                    })
                except Exception as e:
                    results.append({
                        'model': f'Isolation Forest | contamination: {contamination}, n_estimators: {n_estimators}, max_samples: {max_samples}',
                        'contamination': contamination,
                        'n_estimators': n_estimators,
                        'max_samples': max_samples,
                        'precision': 0,
                        'recall': 0,
                        f'f{beta}': 0,
                        'f1': 0,
                        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                        'anomaly_count': -1
                    })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df[f'f{beta}'].idxmax()]

    return results_df, best_row

