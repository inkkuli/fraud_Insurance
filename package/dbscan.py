from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, fbeta_score


def plot_tsne_clusters(df, columns, label_column=None, perplexity=30, random_state=42):
    """
    พล็อต scatter plot หลังใช้ t-SNE ลดมิติของข้อมูลใน df[column]

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names ที่ใช้ลดมิติ
    - label_column: (optional) column name สำหรับ coloring (เช่น cluster หรือ class)
    - perplexity: ค่า hyperparameter ของ t-SNE
    - random_state: สำหรับ reproducibility
    """

    # 1. ตรวจสอบว่า columns อยู่ใน df
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # 2. เตรียมข้อมูล
    X = df[columns].copy()
    X_scaled = StandardScaler().fit_transform(X)

    # 3. ใช้ t-SNE ลดเหลือ 2 มิติ
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_embedded = tsne.fit_transform(X_scaled)

    # 4. สร้าง scatter plot
    plt.figure(figsize=(8, 6))
    if label_column and label_column in df.columns:
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                              c=df[label_column], cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title=label_column)
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7)

    plt.title("t-SNE 2D Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

def dbscan_anomaly(df, eps=0.5, min_samples=5, cols=None):
    """
    ใช้ DBSCAN เพื่อตรวจจับ anomaly ใน DataFrame

    Parameters:
    - df: pandas DataFrame ที่มีข้อมูล
    - eps: ระยะห่างสูงสุดระหว่างจุดสองจุดเพื่อให้เป็นเพื่อนบ้าน
    - min_samples: จำนวนจุดขั้นต่ำใน neighborhood เพื่อให้เป็น core point
    - cols: รายชื่อคอลัมน์ที่ใช้ (ถ้า None จะใช้คอลัมน์ตัวเลขทั้งหมด)

    Returns:
    - df_out: DataFrame เดิมพร้อมคอลัมน์ `anomaly` และ `cluster`
    """
    
    # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    # สเกลข้อมูล
    X_scaled = StandardScaler().fit_transform(df[cols])

    # รัน DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)

    # เพิ่มคอลัมน์ผลลัพธ์
    df_out = df.copy()
    df_out["cluster"] = clusters
    df_out["anomaly"] = (clusters == -1).astype(int)  # -1 = noise

    return df_out

def evaluate_dbscan_performance_with_best(
    df_input,
    df_true,
    label_col,
    pred_col,
    eps_range=np.arange(0.1, 1.1, 0.1),
    min_samples_range=range(3, 11),
    beta=0.5,
    cols=None
):
    y_true = df_true[label_col].astype(int)
    results = []
    

    
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                result_df = dbscan_anomaly(df_input, eps=eps, min_samples=min_samples, cols=cols)
                y_pred = result_df[pred_col]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                results.append({
                    'model' : f'dbscann | eps : {round(eps, 3)} , min_sample : {min_samples}',
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    f'f{beta}': fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'anomaly_count': y_pred.sum()
                })
            except Exception as e:
                results.append({
                    'eps': round(eps, 3),
                    'min_samples': min_samples,
                    'precision': 0,
                    'recall': 0,
                    f'f{beta}': 0,
                    'f1': 0,
                    'anomaly_count': -1
                })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df[f'f{beta}'].idxmax()]
    
    return results_df, best_row