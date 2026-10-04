import pandas as pd
import numpy as np
from typing import Dict, Any, List

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data quality inspection and statistical profiling.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols if total_rows > 0 else 0

    # 1. Missing values & Completeness
    total_missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    completeness_score = max(0.0, round(100.0 - (missing_pct * 2.5), 1))

    # 2. Duplicate rows & Uniqueness
    duplicate_rows_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    uniqueness_score = max(0.0, round(100.0 - (duplicate_pct * 4.0), 1))

    # 3. Column-by-column inspection
    columns_summary = []
    outliers = []
    inconsistencies = []

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        
        # Non-null samples
        clean_series = series.dropna()
        sample_vals = [str(v) if pd.notnull(v) else None for v in clean_series.head(5).tolist()]

        is_bool = pd.api.types.is_bool_dtype(series)
        is_num = pd.api.types.is_numeric_dtype(series) and not is_bool
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        is_cat = not is_num and not is_dt and not is_bool

        col_dict = {
            "name": str(col),
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_cnt,
            "sample_values": sample_vals,
            "is_numeric": is_num,
            "is_categorical": is_cat or is_bool,
            "is_datetime": is_dt,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None
        }

        if is_num and len(clean_series) > 0:
            numeric_cols.append(col)
            min_val = float(clean_series.min())
            max_val = float(clean_series.max())
            mean_val = float(clean_series.mean())
            std_val = float(clean_series.std()) if len(clean_series) > 1 else 0.0

            col_dict["min_value"] = round(min_val, 2)
            col_dict["max_value"] = round(max_val, 2)
            col_dict["mean_value"] = round(mean_val, 2)
            col_dict["std_value"] = round(std_val, 2)

            # Outlier detection via IQR
            q25 = clean_series.quantile(0.25)
            q75 = clean_series.quantile(0.75)
            iqr = q75 - q25

            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
                outlier_count = int(outlier_mask.sum())
                
                if outlier_count > 0:
                    outlier_sample = [round(float(v), 2) for v in clean_series[outlier_mask].head(6).tolist()]
                    outliers.append({
                        "column": str(col),
                        "outlier_count": outlier_count,
                        "outlier_percentage": round((outlier_count / len(clean_series) * 100), 2),
                        "lower_bound": round(float(lower_bound), 2),
                        "upper_bound": round(float(upper_bound), 2),
                        "method": "IQR (1.5 x Interquartile Range)",
                        "sample_outliers": outlier_sample
                    })

            # Check for negative values in naturally non-negative metrics
            lower_name = str(col).lower()
            non_negative_keywords = ["price", "cost", "revenue", "quantity", "age", "days", "count", "income", "amount"]
            if any(k in lower_name for k in non_negative_keywords):
                neg_count = int((clean_series < 0).sum())
                if neg_count > 0:
                    inconsistencies.append({
                        "column": str(col),
                        "issue_type": "Unexpected Negative Values",
                        "description": f"Found {neg_count} negative entries in '{col}', which is conventionally non-negative.",
                        "affected_count": neg_count,
                        "severity": "high" if neg_count > 5 else "medium"
                    })

        elif is_cat:
            categorical_cols.append(col)
            # Check constant or single-value dominance
            if unique_cnt == 1:
                inconsistencies.append({
                    "column": str(col),
                    "issue_type": "Zero Variance / Constant Column",
                    "description": f"Column '{col}' has only 1 distinct value ('{clean_series.iloc[0]}') across all records.",
                    "affected_count": total_rows,
                    "severity": "medium"
                })

        # Check for excessive missingness (> 50%)
        if null_pct >= 50.0:
            inconsistencies.append({
                "column": str(col),
                "issue_type": "Critical Missing Rate (>50%)",
                "description": f"Column '{col}' is missing {null_pct}% of entries ({null_count}/{total_rows}).",
                "affected_count": null_count,
                "severity": "high"
            })

        # Smart Data Type Inconsistency Detection
        data_type_issue = None
        if len(clean_series) >= 3:
            # Case 1: Column is object / string, but values are actually numeric, date, or boolean
            if is_cat:
                sample_text = clean_series.head(50).astype(str).str.strip()
                # Check for boolean ('true'/'false'/'yes'/'no'/'y'/'n')
                lower_vals = sample_text.str.lower()
                bool_set = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0'}
                if lower_vals.isin(bool_set).all() and unique_cnt <= 4:
                    data_type_issue = {
                        "current_type": "text/string",
                        "suggested_type": "boolean",
                        "confidence": 0.95,
                        "reason": f"Column values are exclusively boolean indicators ({', '.join(sample_vals[:3])})."
                    }
                else:
                    # Check for integers: e.g. "12", "-45", "1000"
                    int_pattern = r'^-?\d+$'
                    int_matches = sample_text.str.match(int_pattern).sum()
                    if int_matches / len(sample_text) >= 0.85:
                        data_type_issue = {
                            "current_type": "text/string",
                            "suggested_type": "integer",
                            "confidence": round(int_matches / len(sample_text), 2),
                            "reason": f"{round(int_matches / len(sample_text) * 100)}% of values are valid whole numbers currently stored as text."
                        }
                    else:
                        # Check for float / decimal: e.g. "12.50", "$9.99", "45.0%"
                        cleaned_floats = sample_text.str.replace(r'[\$,%]', '', regex=True)
                        try:
                            parsed_float = pd.to_numeric(cleaned_floats, errors='coerce')
                            valid_floats = int(parsed_float.notnull().sum())
                            if valid_floats / len(sample_text) >= 0.85:
                                data_type_issue = {
                                    "current_type": "text/string",
                                    "suggested_type": "decimal",
                                    "confidence": round(valid_floats / len(sample_text), 2),
                                    "reason": f"{round(valid_floats / len(sample_text) * 100)}% of values are valid numeric decimals stored as text."
                                }
                        except Exception:
                            pass

                    # If not numeric, check for datetime: e.g. "2024-01-15", "05/12/2023"
                    if not data_type_issue:
                        try:
                            # Avoid parsing pure short strings as dates
                            if sample_text.str.len().mean() >= 6:
                                parsed_dt = pd.to_datetime(sample_text, errors='coerce', format='mixed')
                                valid_dt = int(parsed_dt.notnull().sum())
                                if valid_dt / len(sample_text) >= 0.85:
                                    data_type_issue = {
                                        "current_type": "text/string",
                                        "suggested_type": "datetime",
                                        "confidence": round(valid_dt / len(sample_text), 2),
                                        "reason": f"{round(valid_dt / len(sample_text) * 100)}% of values are valid date/time entries formatted as text."
                                    }
                        except Exception:
                            pass

            # Case 2: Column is float64, but all values are integers (no fractional components)
            elif is_num and series.dtype in ['float64', 'float32', 'float']:
                try:
                    is_all_int = ((clean_series % 1) == 0).all()
                    if is_all_int and len(clean_series) > 0:
                        data_type_issue = {
                            "current_type": "decimal/float",
                            "suggested_type": "integer",
                            "confidence": 0.99,
                            "reason": "All values have zero decimal fraction and are whole numbers stored as floating-point."
                        }
                except Exception:
                    pass

        col_dict["data_type_issue"] = data_type_issue
        columns_summary.append(col_dict)

    # Space & Whitespace Issue Detection (Column Names & Text Values)
    cols_with_space_issues = []
    space_issue_count = 0

    for col in df.columns:
        col_str = str(col)
        # Check column name itself
        has_col_name_space_issue = (col_str != col_str.strip()) or ('  ' in col_str)
        # Check values if text
        has_val_space_issue = False
        val_space_cnt = 0
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            s_clean = df[col].dropna().astype(str)
            space_mask = s_clean.str.contains(r'^\s+|\s+$|\s{2,}', regex=True)
            val_space_cnt = int(space_mask.sum())
            if val_space_cnt > 0:
                has_val_space_issue = True

        if has_col_name_space_issue or has_val_space_issue:
            cols_with_space_issues.append({
                "column": col_str,
                "name_has_space": has_col_name_space_issue,
                "values_with_space_count": val_space_cnt
            })
            space_issue_count += val_space_cnt + (1 if has_col_name_space_issue else 0)

    trim_spaces_issues = {
        "has_issues": len(cols_with_space_issues) > 0,
        "affected_columns_count": len(cols_with_space_issues),
        "columns": [c["column"] for c in cols_with_space_issues],
        "details": cols_with_space_issues,
        "total_irregular_occurrences": space_issue_count
    }

    # 4. Consistency & Validity Scores
    total_issues = len(inconsistencies)
    consistency_score = max(0.0, round(100.0 - (total_issues * 8.0), 1))

    total_outliers_count = sum(o["outlier_count"] for o in outliers)
    outlier_pct = round((total_outliers_count / (total_rows * len(numeric_cols)) * 100), 2) if (total_rows > 0 and numeric_cols) else 0.0
    validity_score = max(0.0, round(100.0 - (outlier_pct * 3.0), 1))

    # Overall Quality Score
    overall_quality_score = round(
        (completeness_score * 0.35) +
        (uniqueness_score * 0.25) +
        (consistency_score * 0.20) +
        (validity_score * 0.20),
        1
    )

    # 5. Correlation matrix for numeric features
    correlations = {}
    if len(numeric_cols) >= 2:
        try:
            corr_df = df[numeric_cols].corr().fillna(0)
            for r_col in corr_df.index:
                correlations[str(r_col)] = {str(c_col): round(float(corr_df.loc[r_col, c_col]), 3) for c_col in corr_df.columns}
        except Exception:
            correlations = {}

    # 6. Important Variables (ranked by statistical variance & correlation influence)
    important_variables = []
    for c in numeric_cols:
        series = df[c].dropna()
        if len(series) > 1:
            variance = float(series.var())
            cv = float(series.std() / (series.mean() if series.mean() != 0 else 1.0))
            
            # Find strongest correlation
            top_corr_partner = None
            max_corr_val = 0.0
            if c in correlations:
                for other_c, val in correlations[c].items():
                    if other_c != c and abs(val) > abs(max_corr_val):
                        max_corr_val = val
                        top_corr_partner = other_c

            important_variables.append({
                "column": str(c),
                "type": "numeric",
                "variance": round(variance, 2),
                "coefficient_of_variation": round(abs(cv), 2),
                "strongest_correlation": {
                    "partner_column": top_corr_partner,
                    "correlation_coefficient": round(max_corr_val, 3)
                } if top_corr_partner else None,
                "importance_score": round(min(100.0, (abs(cv) * 20.0) + (abs(max_corr_val) * 40.0)), 1)
            })

    # Sort important variables by importance_score descending
    important_variables.sort(key=lambda x: x["importance_score"], reverse=True)

    # 7. Trends & Patterns
    trends_and_patterns = []
    for c in numeric_cols[:4]:
        series = df[c].dropna()
        if len(series) >= 10:
            skew = float(series.skew())
            skew_desc = "Right-skewed (concentrated at lower end with high outliers)" if skew > 1.0 else (
                "Left-skewed (concentrated at higher end with low outliers)" if skew < -1.0 else "Approximately symmetrical"
            )
            trends_and_patterns.append({
                "variable": str(c),
                "pattern_type": "Distribution Skew",
                "metric_value": round(skew, 2),
                "interpretation": skew_desc
            })

    # 8. Next Investigations (data quality recommendations based on findings)
    next_investigations = []
    for inc in inconsistencies[:3]:
        next_investigations.append(f"{inc['issue_type']} in '{inc['column']}': {inc['description']}")
    if len(outliers) > 0:
        next_investigations.append(f"Investigate outliers in '{outliers[0]['column']}' ({outliers[0]['outlier_count']} values outside IQR bounds).")
    if total_missing_cells > 0:
        most_null = sorted(
            [(str(col), int(df[col].isnull().sum())) for col in df.columns if df[col].isnull().sum() > 0],
            key=lambda x: x[1], reverse=True
        )
        if most_null:
            next_investigations.append(f"Address {most_null[0][1]:,} missing values in '{most_null[0][0]}' before analysis.")
    if duplicate_rows_count > 0:
        next_investigations.append(f"Remove {duplicate_rows_count:,} duplicate rows to ensure analysis accuracy.")

    return {
        "overall_quality_score": overall_quality_score,
        "completeness_score": completeness_score,
        "uniqueness_score": uniqueness_score,
        "consistency_score": consistency_score,
        "validity_score": validity_score,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cells_percentage": missing_pct,
        "duplicate_rows_count": duplicate_rows_count,
        "duplicate_rows_percentage": duplicate_pct,
        "columns_summary": columns_summary,
        "outliers": outliers,
        "inconsistencies": inconsistencies,
        "correlations": correlations,
        "important_variables": important_variables[:6],
        "trends_and_patterns": trends_and_patterns,
        "next_investigations": next_investigations,
        "trim_spaces_issues": trim_spaces_issues,
    }
