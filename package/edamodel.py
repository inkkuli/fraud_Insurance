from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

def plot_pca_anomaly(result_df, anomaly_col='anomaly'):
    """
    สร้างกราฟ PCA 2D โดยเลือกฟีเจอร์หลัก และระบุคอลัมน์ที่ใช้แบ่งกลุ่ม anomaly
    Parameters:
        result_df (pd.DataFrame): ข้อมูลที่มี features และคอลัมน์ anomaly
        anomaly_col (str): ชื่อคอลัมน์ที่ใช้แบ่งสี (default = 'anomaly')
    """
    # เลือกฟีเจอร์สำหรับ PCA
    feature_cols = [
        'months_as_customer',
        'injury_claim', 'property_claim', 'vehicle_claim',
        'policy_annual_premium'
    ]

    # ตรวจสอบคอลัมน์
    missing = [col for col in feature_cols + [anomaly_col] if col not in result_df.columns]
    if missing:
        raise ValueError(f"Missing columns in result_df: {missing}")

    # เตรียมข้อมูล
    X = result_df[feature_cols]
    y = result_df[anomaly_col]

    # สเกลข้อมูล
    X_scaled = StandardScaler().fit_transform(X)

    # ทำ PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # รวมเป็น DataFrame สำหรับ plot
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'anomaly': y
    })

    # Plot
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(pca_df['anomaly'].unique())
    colors = plt.cm.Set1.colors  # ใช้ color map

    for i, val in enumerate(unique_labels):
        subset = pca_df[pca_df['anomaly'] == val]
        plt.scatter(subset['PC1'], subset['PC2'],
                    c=[colors[i % len(colors)]], label=str(val), alpha=0.6, edgecolor='k')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA - Colored by {anomaly_col}')
    plt.legend(title=anomaly_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()