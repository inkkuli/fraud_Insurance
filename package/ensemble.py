
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import fbeta_score

def weighted_voting_from_scores(df,df_result_if, df_result_dbscan,df_result_mh,f05_if, f05_db, f05_md, threshold=0.5):
    """
    คำนวณ weighted voting สำหรับ anomaly detection
    
    Parameters:
    - df : pd.DataFrame ที่มีคอลัมน์ anomaly_if, anomaly_db, anomaly_md (0 หรือ 1)
    - f1_if, f1_db, f1_md : float — f1-score ของแต่ละโมเดล
    - threshold : float — ค่า cutoff สำหรับถือว่าเป็น anomaly (default = 0.5)

    Returns:
    - df : DataFrame เดิมที่เพิ่มคอลัมน์ score_voted และ anomaly_voted
    """
    # 1. Normalize weights
    total = f05_if + f05_db + f05_md
    w_if = f05_if / total
    w_db = f05_db / total
    w_md = f05_md / total

    df['anomaly_if'] = df_result_if['anomaly']
    df['anomaly_db'] = df_result_dbscan['anomaly']
    df['anomaly_md'] = df_result_mh['anomaly']

    # 2. Weighted score
    df['score_voted'] = (
        w_if * df['anomaly_if'] +
        w_db * df['anomaly_db'] +
        w_md * df['anomaly_md']
    )

    # 3. Apply threshold
    df['anomaly_voted'] = (df['score_voted'] >= threshold).astype(int)

    return df

def stacking_xgb(df_result_if, df_result_dbscan, df_result_mh, df_true, test_size=0.3, random_state=42):
    """
    รวม feature จาก base models และฝึก meta-model ด้วย XGBoost
    
    Parameters:
    - df_if: DataFrame with 'if_score', 'if_label'
    - df_dbscan: DataFrame with 'dbscan_label'
    - df_mahal: DataFrame with 'mahal_score', 'mahal_label'
    - df_true: DataFrame with 'label' (ground truth)
    - test_size: float (default 0.3) ขนาดของ test split
    - random_state: int (default 42)
    
    Returns:
    - model: XGBClassifier ที่ถูกฝึกแล้ว
    - X_test, y_test: ข้อมูล test set
    - y_pred: คำทำนายจาก meta-model
    """
    # 1. รวมฟีเจอร์
    X_stack = pd.concat([
    df_result_if[['anomaly_score', 'anomaly']].rename(columns={
        'anomaly_score': 'if_score', 'anomaly': 'if_label'
    }).reset_index(drop=True),
    df_result_dbscan[['anomaly']].rename(columns={
        'anomaly': 'dbscan_label'
    }).reset_index(drop=True),
    df_result_mh[['mahalanobis_distance', 'anomaly']].rename(columns={
        'mahalanobis_distance': 'mh_score', 'anomaly': 'mh_label'
    }).reset_index(drop=True)
], axis=1)

    y_stack = df_true['fraud_reported'].reset_index(drop=True)

    # 2. แบ่ง Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=test_size, random_state=random_state, stratify=y_stack
    )

    # 3. ฝึก XGBoost meta model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train)

    # 4. ทำนายและแสดงผล
    y_pred = model.predict(X_test)
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    print(f"🎯 F0.5 score: {f05:.4f}")


    return model, X_test, y_test, y_pred , f05