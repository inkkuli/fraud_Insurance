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

