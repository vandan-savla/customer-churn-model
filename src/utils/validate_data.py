from typing import List, Tuple

import pandas as pd


def _append_if(condition: bool, failures: List[str], message: str) -> None:
    if condition:
        failures.append(message)


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the Telco churn dataset using explicit pandas checks.

    This avoids depending on deprecated Great Expectations APIs at runtime while
    keeping the same business validation intent.
    """
    print("Starting data validation...")
    failures: List[str] = []

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    _append_if(bool(missing_columns), failures, f"missing_columns:{missing_columns}")

    if missing_columns:
        return False, failures

    _append_if(df["customerID"].isna().any(), failures, "customerID_nulls")
    _append_if(~df["gender"].isin(["Male", "Female"]).all(), failures, "gender_invalid_values")
    _append_if(~df["Partner"].isin(["Yes", "No"]).all(), failures, "Partner_invalid_values")
    _append_if(~df["Dependents"].isin(["Yes", "No"]).all(), failures, "Dependents_invalid_values")
    _append_if(~df["PhoneService"].isin(["Yes", "No"]).all(), failures, "PhoneService_invalid_values")
    _append_if(
        ~df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all(),
        failures,
        "Contract_invalid_values",
    )
    _append_if(
        ~df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all(),
        failures,
        "InternetService_invalid_values",
    )

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total_raw = df["TotalCharges"].astype(str).str.strip()
    total = pd.to_numeric(total_raw.mask(total_raw == "", other=None), errors="coerce")

    _append_if(tenure.isna().any(), failures, "tenure_non_numeric")
    _append_if(monthly.isna().any(), failures, "MonthlyCharges_non_numeric")
    total_invalid_mask = total.isna() & total_raw.ne("")
    _append_if(total_invalid_mask.any(), failures, "TotalCharges_non_numeric")

    _append_if((tenure < 0).any(), failures, "tenure_negative")
    _append_if((tenure > 120).any(), failures, "tenure_out_of_range")
    _append_if((monthly < 0).any(), failures, "MonthlyCharges_negative")
    _append_if((monthly > 200).any(), failures, "MonthlyCharges_out_of_range")
    _append_if((total < 0).any(), failures, "TotalCharges_negative")
    _append_if(tenure.isna().any(), failures, "tenure_nulls")
    _append_if(monthly.isna().any(), failures, "MonthlyCharges_nulls")

    valid_pair_mask = total.notna() & monthly.notna()
    pair_fail_rate = ((total[valid_pair_mask] < monthly[valid_pair_mask]).mean() if valid_pair_mask.any() else 0.0)
    _append_if(pair_fail_rate > 0.05, failures, "TotalCharges_less_than_MonthlyCharges_too_often")

    if failures:
        print(f"Data validation FAILED: {len(failures)} checks failed")
        print(f"   Failed expectations: {failures}")
        return False, failures

    print("Data validation PASSED")
    return True, failures
