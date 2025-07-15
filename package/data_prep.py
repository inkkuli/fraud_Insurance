import pandas as pd
def check_df_info(df):
    total_rows = len(df)
    null_counts = df.isnull().sum()
    outlier_counts = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            outlier_counts.append(len(outliers))
        else:
            outlier_counts.append(None)

    info_table = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'null_count': null_counts.values,
        'null_percent': (null_counts.values / total_rows * 100).round(2),
        'total_rows': total_rows,
        'outlier_count': outlier_counts
    })

    return info_table

def check_duplicates(df, subset=None, keep='first', show_sample=True):
    """
    ตรวจสอบค่าซ้ำใน DataFrame
    - subset: ระบุคอลัมน์ที่ต้องการตรวจซ้ำ (ถ้าไม่ระบุ จะตรวจทั้งแถว)
    - keep: เลือกว่าจะเก็บแถวไหน ('first', 'last', False)
    - show_sample: แสดงตัวอย่างค่าซ้ำหรือไม่

    Return: dict สรุปผล
    """
    dup_mask = df.duplicated(subset=subset, keep=keep)
    dup_df = df[dup_mask]

    result = {
        'total_duplicates': dup_mask.sum(),
        'duplicate_rows': dup_df if show_sample else None,
        'columns_checked': subset if subset else 'all columns'
    }

    return result


def filter_and_convert_columns(df):
    # แยกคอลัมน์ตามชนิดข้อมูลที่ต้องการ
    int_columns = [
        'months_as_customer',
        'injury_claim',
        'property_claim', 'vehicle_claim'
    ]

    float_columns = ['policy_annual_premium']

    all_columns = int_columns + float_columns

    # คัดเฉพาะคอลัมน์ที่มีอยู่จริงใน df
    existing_columns = [col for col in all_columns if col in df.columns]

    # เลือกเฉพาะคอลัมน์ที่ต้องการ
    df = df[existing_columns].copy()

    # แปลงชนิดข้อมูล
    df[int_columns] = df[int_columns].fillna(0).astype(int)
    df[float_columns] = df[float_columns].fillna(0.0).astype(float)

    return df

def clean_dataframe(df, outlier_cols=None):
    """
    ลบ missing values, duplicated rows, และ outliers (แบบ IQR) จาก DataFrame
    - outlier_cols: รายชื่อคอลัมน์ที่ต้องการตรวจ outlier (ถ้า None จะใช้ numeric ทั้งหมด)
    """
    df_clean = df.copy()

    # 1. ลบ NaN
    df_clean.dropna(inplace=True)

    # 2. ลบ duplicated rows
    df_clean.drop_duplicates(inplace=True)

    # 3. ลบ outliers (ใช้ IQR)
    # if outlier_cols is None:
    #     outlier_cols = df_clean.select_dtypes(include=['int', 'float']).columns.tolist()

    # for col in outlier_cols:
    #     q1 = df_clean[col].quantile(0.25)
    #     q3 = df_clean[col].quantile(0.75)
    #     iqr = q3 - q1
    #     lower = q1 - 1.5 * iqr
    #     upper = q3 + 1.5 * iqr
    #     df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean

def convert_fraud_column(df, column='fraud_reported'):
    """
    แปลงค่าคอลัมน์ 'fraud_reported' จาก 'Y' → 1 และ 'N' → 0
    Parameters:
        df (pd.DataFrame): DataFrame ที่ต้องการแปลง
        column (str): ชื่อคอลัมน์ที่ต้องการแปลง
    Returns:
        pd.DataFrame: DataFrame ที่มีการแปลงค่าคอลัมน์เรียบร้อยแล้ว
    """
    df = df.copy()
    df["fraud_reported"] = df["fraud_reported"].map({'Y': 1, 'N': 0})
    return df

def merge_model_results(df, result_df):
    """
    รวมคอลัมน์ผลลัพธ์จาก model (result_df) เข้ากับ df โดย:
    - ใช้ df เป็นหลัก
    - รวมเฉพาะคอลัมน์ที่มีใน result_df และอยู่ในรายการเป้าหมาย
    Parameters:
        df (pd.DataFrame): ข้อมูลต้นฉบับ
        result_df (pd.DataFrame): ข้อมูลผลลัพธ์จาก model
    Returns:
        pd.DataFrame: DataFrame ใหม่ที่รวมผลลัพธ์เฉพาะคอลัมน์ที่ตรง
    """
    target_columns = ['mahalanobis_distance', 'anomaly', 'p_value', 'anomaly_pvalue']
    available_columns = [col for col in target_columns if col in result_df.columns]

    # เช็คว่า row ตรงกันไหม
    if len(df) != len(result_df):
        raise ValueError("จำนวนแถวของ df และ result_df ต้องเท่ากัน")

    # รวมข้อมูล
    df_merged = pd.concat([df.reset_index(drop=True), result_df[available_columns].reset_index(drop=True)], axis=1)
    return df_merged