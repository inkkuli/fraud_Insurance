from pingouin import multivariate_normality
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix

def check_multivariate(df):
    hz, pval, normal = multivariate_normality(df, alpha=0.05)
    result = {f"HZ = {hz:.3f}, p-value = {pval:.3f}, normal = {normal}"}
    return result

def check_all_columns_normality(df):
    """
    ตรวจสอบการแจกแจงปกติของทุกคอลัมน์ที่เป็นตัวเลขใน df
    - ใช้ Shapiro-Wilk Test
    - แสดง Q-Q Plot
    """
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            print(f"⚠️ Column '{col}' has too few data points for testing.\n")
            continue

        # Shapiro-Wilk Test
        stat, p = stats.shapiro(data)
        print(f"{col} — Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p:.4f}")
        if p > 0.05:
            print("✅ Likely normal distribution (fail to reject H0)")
        else:
            print("❌ Likely NOT normal distribution (reject H0)")

        # Q-Q Plot
        plt.figure(figsize=(5, 5))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: {col}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.grid(True)
        plt.show()

def check_normality_anderson(df):
    """
    ตรวจสอบการแจกแจงแบบปกติของทุกคอลัมน์ตัวเลขใน DataFrame
    โดยใช้ Anderson-Darling Test และ Q-Q Plot
    """
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 5:
            print(f"⚠️ Column '{col}' มีข้อมูลน้อยเกินไปสำหรับการทดสอบ\n")
            continue

        # Anderson-Darling Test
        result = stats.anderson(data, dist='norm')
        print(
            f"📊 Anderson-Darling Test for '{col}': A² = {result.statistic:.4f}")
        print("Critical values (sig level %):")
        for sl, cv in zip(result.significance_level, result.critical_values):
            status = "❌ Reject H0" if result.statistic > cv else "✅ Fail to Reject H0"
            print(f"  - {sl}%: {cv:.4f} → {status}")
        print()

        # Q-Q Plot
        plt.figure(figsize=(5, 5))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: {col}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.grid(True)
        plt.show()

def normalize_column(df, column, method='yeo-johnson'):
    """
    แปลงคอลัมน์ให้เข้าใกล้ normal distribution
    method: 'log', 'sqrt', 'boxcox', 'yeo-johnson'
    คืนค่า: Series ใหม่ที่แปลงแล้ว
    """
    data = df[column].dropna()

    if method == 'log':
        if (data < 0).any():
            raise ValueError("Log ใช้กับค่าบวกเท่านั้น")
        return np.log1p(df[column])  # log(x + 1)

    elif method == 'sqrt':
        if (data < 0).any():
            raise ValueError("Sqrt ใช้กับค่าบวกเท่านั้น")
        return np.sqrt(df[column])

    elif method == 'boxcox':
        if (data <= 0).any():
            raise ValueError("Box-Cox ใช้ได้กับค่ามากกว่า 0 เท่านั้น")
        transformed, _ = boxcox(data)
        df_result = df[column].copy()
        df_result.loc[data.index] = transformed
        return df_result

    elif method == 'yeo-johnson':
        pt = PowerTransformer(method='yeo-johnson')
        reshaped = df[[column]].copy()
        reshaped[column] = reshaped[column].fillna(0)  # หรือกำหนดเอง
        transformed = pt.fit_transform(reshaped)
        return pd.Series(transformed.flatten(), index=df.index, name=column + '_transformed')

    else:
        raise ValueError(f"ไม่รู้จัก method '{method}' — ใช้ได้: 'log', 'sqrt', 'boxcox', 'yeo-johnson'")

def check_determinant(df):
    cov = np.cov(df.T)
    det = np.linalg.det(cov)
    if np.isclose(det, 0.0):
        print("❌ Covariance matrix is **singular** (determinant ≈ 0)")
    else:
        print(
            f"✅ Covariance matrix is **non-singular** (determinant = {det:.6f})")
    return det

def robust_mahalanobis_anomaly(X_input, threshold_percentile=0.8, return_distance=False, random_state=42):
    """
    Robust Mahalanobis Distance anomaly detection (Z-score scaling done inside)

    Parameters:
    - X_input: pandas DataFrame (raw scale)
    - threshold_percentile: float, percentile cutoff (default=0.975)
    - return_distance: if True, return also the threshold
    - random_state: int, for reproducibility

    Returns:
    - result_df: DataFrame (original scale) + mahalanobis_distance + anomaly
    """
    if not isinstance(X_input, pd.DataFrame):
        raise ValueError("X_input must be a pandas DataFrame")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    # ✅ Fix randomness
    mcd = MinCovDet(random_state=random_state).fit(X_scaled)
    distances = mcd.mahalanobis(X_scaled)

    threshold = np.percentile(distances, threshold_percentile * 100)
    anomalies = (distances > threshold).astype(int)

    result_df = X_input.copy()
    result_df["mahalanobis_distance"] = distances
    result_df["anomaly"] = anomalies

    if return_distance:
        return result_df, threshold
    else:
        return result_df


def add_pvalue_to_result(result_df, alpha=0.05):
    """
    เพิ่ม p-value และ anomaly_pvalue โดยไม่ต้องระบุ n_features
    จะตรวจหา df อัตโนมัติจากคอลัมน์ใน result_df

    Parameters:
        result_df: DataFrame ที่ได้จาก robust_mahalanobis_anomaly()
        alpha: ค่าความเชื่อมั่น (default = 0.05)

    Returns:
        result_df: DataFrame เดิมที่เพิ่ม 'p_value' และ 'anomaly_pvalue'
    """
    if "mahalanobis_distance" not in result_df.columns:
        raise ValueError("Column 'mahalanobis_distance' not found in input DataFrame.")

    # คำนวณ d^2
    d_squared = result_df["mahalanobis_distance"] ** 2

    # หา feature columns = all - known outputs
    ignore_cols = {"mahalanobis_distance", "anomaly", "p_value", "anomaly_pvalue"}
    feature_cols = [col for col in result_df.columns if col not in ignore_cols]
    df = len(feature_cols)

    # คำนวณ p-value และ anomaly จาก cutoff
    p_values = 1 - chi2.cdf(d_squared, df=df)
    result_df["p_value"] = p_values
    result_df["anomaly_pvalue"] = (p_values < alpha).astype(int)

    return result_df


def evaluate_percentile_alpha_performance_with_best(
    X_input,
    df_true,
    label_col='fraud_reported',
    percentile=np.arange(0.90, 0.991, 0.01),
    alphas=np.arange(0.01, 0.201, 0.025),
    beta=0.5 
):
    if df_true[label_col].dtype == object:
        y_true = df_true[label_col].map({'Y': 1, 'N': 0}).astype(int)
    else:
        y_true = df_true[label_col].astype(int)

    percentile_rows = []
    for q in percentile:
        result_df = robust_mahalanobis_anomaly(X_input, threshold_percentile=q)
        y_pred = result_df['anomaly']

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        percentile_rows.append({
            'percentile': f'Mahalanobis | percentile : {round(q, 3)}',
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            f'f{beta}': fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'anomaly_count': y_pred.sum()
        })

    percentile_df = pd.DataFrame(percentile_rows)
    best_q_row = percentile_df.loc[percentile_df[f'f{beta}'].idxmax()]

    # สำหรับ alpha part (fixed percentile = 0.975)
    result_df = robust_mahalanobis_anomaly(X_input, threshold_percentile=0.975)
    result_df = add_pvalue_to_result(result_df, alpha=0.05)

    alpha_rows = []
    for alpha in alphas:
        y_pred = (result_df["p_value"] < alpha).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        alpha_rows.append({
            'alpha': f'Mahalanobis | alpha :{round(alpha, 3)}',
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            f'f{beta}': fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'anomaly_count': y_pred.sum()
        })

    alpha_df = pd.DataFrame(alpha_rows)
    best_a_row = alpha_df.loc[alpha_df[f'f{beta}'].idxmax()]

    return percentile_df, alpha_df, best_q_row, best_a_row
