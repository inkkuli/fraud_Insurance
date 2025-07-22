
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import fbeta_score

def weighted_voting_from_scores(df_true,df_result_if, df_result_dbscan,df_result_mh,pred_col_if,pred_col_dbscan,pred_col_md,f05_if, f05_db, f05_md, threshold=0.5):
    """
    คำนวณ weighted voting สำหรับ anomaly detection
    
    Parameters:
    - df_true : pd.DataFrame ที่มีคอลัมน์ anomaly_if, anomaly_db, anomaly_md (0 หรือ 1)
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

    df_voted = df_true.copy()

    df_voted['anomaly_if'] = df_result_if[pred_col_if]
    df_voted['anomaly_db'] = df_result_dbscan[pred_col_dbscan]
    df_voted['anomaly_md'] = df_result_mh[pred_col_md]

    # 2. Weighted score
    df_voted['score_voted'] = (
        w_if * df_voted['anomaly_if'] +
        w_db * df_voted['anomaly_db'] +
        w_md * df_voted['anomaly_md']
    )

    # 3. Apply threshold
    df_voted['anomaly_voted'] = (df_voted['score_voted'] >= threshold).astype(int)

    return df_voted


def stacking_xgb(
    df_result_if, df_result_dbscan, df_result_mh, df_true,
    if_score_col, if_label_col,
    dbscan_label_col,
    mh_score_col, mh_label_col,
    true_label_col,
    test_size=0.3, random_state=42
):
    """
    รวม feature จาก base models และฝึก meta-model ด้วย XGBoost

    Parameters:
    - df_result_if: DataFrame ที่มี if_score_col และ if_label_col
    - df_result_dbscan: DataFrame ที่มี dbscan_label_col
    - df_result_mh: DataFrame ที่มี mh_score_col และ mh_label_col
    - df_true: DataFrame ที่มี true_label_col (ground truth)
    - *_col: ชื่อคอลัมน์แต่ละประเภท
    - test_size: ขนาดของ test split
    - random_state: ค่า random seed

    Returns:
    - model: XGBClassifier ที่ฝึกแล้ว
    - X_test, y_test: test set
    - y_pred: คำทำนาย
    - f05: F0.5 score
    """
    
    # 1. รวมฟีเจอร์จาก base models
    X_stack = pd.concat([
        df_result_if[[if_score_col, if_label_col]].rename(columns={
            if_score_col: 'if_score',
            if_label_col: 'if_label'
        }).reset_index(drop=True),

        df_result_dbscan[[dbscan_label_col]].rename(columns={
            dbscan_label_col: 'dbscan_label'
        }).reset_index(drop=True),

        df_result_mh[[mh_score_col, mh_label_col]].rename(columns={
            mh_score_col: 'mh_score',
            mh_label_col: 'mh_label'
        }).reset_index(drop=True)
    ], axis=1)

    # 2. label
    y_stack = df_true[true_label_col].reset_index(drop=True)

    # 3. แบ่ง train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y_stack, test_size=test_size, random_state=random_state, stratify=y_stack
    )

    # 4. ฝึก XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train)

    # 5. ทำนายและประเมินผล
    y_pred = model.predict(X_test)
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    print(f"🎯 F0.5 score: {f05:.4f}")

    return model, X_test, y_test, y_pred, f05
