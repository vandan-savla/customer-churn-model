import pandas as pd


def _map_binary_series(series: pd.Series) -> pd.Series:
    vals = list(pd.Series(series.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return series.map({"No": 0, "Yes": 1}).astype("Int64")

    if valset == {"Male", "Female"}:
        return series.map({"Female": 0, "Male": 1}).astype("Int64")

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return series.astype(str).map(mapping).astype("Int64")

    return series


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    obj_cols = [column for column in df.select_dtypes(include=["object"]).columns if column != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"   Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    binary_cols = [column for column in obj_cols if df[column].dropna().nunique() == 2]
    multi_cols = [column for column in obj_cols if df[column].dropna().nunique() > 2]

    print(f"   Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    for column in binary_cols:
        original_dtype = df[column].dtype
        df[column] = _map_binary_series(df[column].astype(str))
        print(f"      {column}: {original_dtype} -> binary (0/1)")

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    if multi_cols:
        print(f"   Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      Created {new_features} new features from {len(multi_cols)} categorical columns")

    for column in binary_cols:
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df
