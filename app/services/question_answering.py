import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def _normalize_str(s: str) -> str:
    """Normalizes string by removing non-alphanumeric characters and lowering."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _extract_column_matches(query: str, df: pd.DataFrame) -> List[Tuple[str, int, str]]:
    """
    Dynamically identifies which columns in df are mentioned in the query.
    Returns list of (column_name, match_score, col_type) sorted by match quality.
    Works for ANY dataset and ANY column naming convention.
    """
    q_lower = query.lower()
    q_tokens = set(re.findall(r'[a-z0-9]+', q_lower))
    matches = []

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        col_clean = re.sub(r'[^a-z0-9]', ' ', col_lower).strip()
        col_tokens = set(col_clean.split())
        
        is_num = pd.api.types.is_numeric_dtype(df[col])
        is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
        col_type = "numeric" if is_num else ("datetime" if is_dt else "categorical")

        score = 0
        # 1. Exact string in query
        if col_lower in q_lower or col_clean in q_lower:
            score = 100 + len(col_clean)
        # 2. Normalized alphanumeric match
        elif _normalize_str(col_str) in _normalize_str(query):
            score = 90 + len(_normalize_str(col_str))
        # 3. All column tokens appear in query
        elif col_tokens and col_tokens.issubset(q_tokens):
            score = 80 + len(col_tokens)
        # 4. Partial token overlap
        elif col_tokens:
            common = col_tokens.intersection(q_tokens)
            valid_common = [w for w in common if len(w) > 2]
            if valid_common:
                score = 40 + len(valid_common) * 10

        if score > 0:
            matches.append((col, score, col_type))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def _generate_procedure_steps(query: str, df: pd.DataFrame, col_matches: List[Tuple[str, int, str]]) -> Optional[str]:
    """
    When query requests a procedure, instructions, or steps:
    Returns strictly numbered sequential steps with ZERO filler, introductions, conclusions, or narrative.
    """
    q_lower = query.lower().strip()
    is_procedure = any(k in q_lower for k in [
        "step", "steps", "procedure", "how to", "instructions", "workflow", "process to", "guide to"
    ])
    if not is_procedure:
        return None

    target_col = col_matches[0][0] if col_matches else None
    col_type = col_matches[0][2] if col_matches else "general"

    # Cleaning missing values
    if any(k in q_lower for k in ["missing", "null", "impute", "empty"]):
        col_name = f"'{target_col}'" if target_col else "the target column"
        if col_type == "numeric":
            return (
                f"1. Navigate to the Data Cleaning stage and select {col_name}.\n"
                f"2. Inspect the missing value percentage and distribution spread.\n"
                f"3. Select 'Fill Missing' and choose median imputation to resist skewness, or mean if distribution is symmetric.\n"
                f"4. Apply the imputation decision to generate a preview.\n"
                f"5. Save and commit the cleaning plan to update dataset health metrics."
            )
        else:
            return (
                f"1. Open the Data Cleaning interface and locate {col_name}.\n"
                f"2. Review null count and category frequencies.\n"
                f"3. Select mode imputation for most frequent category, or fill with a constant label such as 'Unknown'.\n"
                f"4. Verify the row impact in the preview table.\n"
                f"5. Commit the transformation to execute the cleaning pipeline."
            )

    # Outlier handling
    if any(k in q_lower for k in ["outlier", "extreme", "iqr", "anomaly"]):
        col_name = f"'{target_col}'" if target_col else "the numeric column"
        return (
            f"1. Open Data Cleaning and identify {col_name}.\n"
            f"2. Review the IQR lower and upper boundary thresholds.\n"
            f"3. Choose 'Clip Outliers' to bound extreme tail values to IQR fences, or 'Remove Rows' if values represent corrupt data.\n"
            f"4. Check the before-and-after variance in the audit preview.\n"
            f"5. Apply changes to execute the cleaning plan."
        )

    # Deduplication
    if any(k in q_lower for k in ["duplicate", "dedup", "identical"]):
        return (
            f"1. Open the Data Cleaning panel.\n"
            f"2. Enable the 'Remove Duplicate Rows' toggle.\n"
            f"3. Inspect the projected row count reduction in the summary.\n"
            f"4. Confirm the removal action.\n"
            f"5. Commit the plan to generate the audit log."
        )

    # General transformation / export / workflow
    if any(k in q_lower for k in ["export", "report", "download", "pdf"]):
        return (
            f"1. Complete data inspection and any desired cleaning decisions.\n"
            f"2. Navigate to the Data Intelligence and Report section.\n"
            f"3. Review the executive metrics and automated audit trail.\n"
            f"4. Click 'Download Company PDF Report'.\n"
            f"5. Save the generated compliance document locally."
        )

    # Default general procedure
    target = f" on '{target_col}'" if target_col else ""
    return (
        f"1. Inspect column data types and distribution metrics in Data Inspection.\n"
        f"2. Configure cleaning rules{target} in Data Cleaning.\n"
        f"3. Review calculated quality scores in AI Intelligence.\n"
        f"4. Select chart architectures in Visualization and Insights.\n"
        f"5. Export the verified report in Data Intelligence and Report."
    )


def answer_dataset_question(df: pd.DataFrame, question: str, quality_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Answers natural language questions about the currently uploaded dataset dynamically.
    Inspects actual columns, rows, types, and values.
    Calculates specific numbers, lists, aggregations, rankings, or comparisons directly from df.
    Contains ZERO hardcoded domains, datasets, or column names.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    q_lower = question.lower().strip()

    # Guard: empty dataset
    if total_rows == 0:
        return {
            "question": question,
            "answer": "The uploaded dataset is empty (0 rows). No records are available to analyze.",
            "evidence": {"total_rows": 0, "total_columns": total_cols},
            "takeaway": "Please upload a dataset containing at least one record."
        }

    col_matches = _extract_column_matches(question, df)
    cat_matches = [m[0] for m in col_matches if m[2] == "categorical"]
    num_matches = [m[0] for m in col_matches if m[2] == "numeric"]

    # -------------------------------------------------------------------------
    # 0. Steps-Only Requirement for Procedures / Instructions
    # -------------------------------------------------------------------------
    procedure_text = _generate_procedure_steps(question, df, col_matches)
    if procedure_text:
        return {
            "question": question,
            "answer": procedure_text,
            "evidence": {
                "procedure_type": "sequential_steps",
                "target_feature": str(col_matches[0][0]) if col_matches else "dataset"
            },
            "takeaway": "Follow the numbered sequence to execute this action."
        }

    # -------------------------------------------------------------------------
    # 1. Dataset Dimension & Quality Questions
    # -------------------------------------------------------------------------
    if any(k in q_lower for k in ["how many rows", "number of rows", "total rows", "row count", "how many records", "size of dataset"]):
        return {
            "question": question,
            "answer": f"The uploaded dataset contains exactly {total_rows:,} rows across {total_cols} columns ({total_rows * total_cols:,} total cells).",
            "evidence": {"total_rows": total_rows, "total_columns": total_cols},
            "takeaway": f"Total volume: {total_rows:,} records."
        }

    if any(k in q_lower for k in ["how many columns", "number of columns", "list of columns", "what columns", "column names", "available fields"]):
        cols_list = list(df.columns)
        return {
            "question": question,
            "answer": f"The dataset contains {total_cols} columns:\n" + "\n".join([f"• {c} ({df[c].dtype})" for c in cols_list]),
            "evidence": {"columns": [str(c) for c in cols_list], "count": total_cols},
            "takeaway": f"{total_cols} features available."
        }

    if any(k in q_lower for k in ["missing", "null", "empty", "incomplete", "nan"]):
        missing_s = df.isnull().sum()
        cols_with_null = missing_s[missing_s > 0].sort_values(ascending=False)
        tot_missing = int(missing_s.sum())
        if cols_with_null.empty:
            return {
                "question": question,
                "answer": "There are zero missing or null values in this dataset. Every feature is 100% complete across all records.",
                "evidence": {"total_missing": 0, "columns_with_missing": {}},
                "takeaway": "Data completeness is at optimal 100%."
            }
        top_nulls = {str(col): int(cnt) for col, cnt in cols_with_null.items()}
        lines = [f"• {c}: {cnt:,} missing ({cnt/total_rows*100:.1f}%)" for c, cnt in list(top_nulls.items())[:15]]
        return {
            "question": question,
            "answer": f"Total missing cells: {tot_missing:,}. Columns with missing values:\n" + "\n".join(lines),
            "evidence": {"total_missing": tot_missing, "missing_by_column": top_nulls},
            "takeaway": f"Highest missingness is in '{cols_with_null.index[0]}' ({cols_with_null.iloc[0]:,} nulls)."
        }

    if any(k in q_lower for k in ["duplicate", "identical rows", "repeated rows"]):
        dup_count = int(df.duplicated().sum())
        return {
            "question": question,
            "answer": f"The dataset contains {dup_count:,} duplicate row(s) ({dup_count/total_rows*100:.2f}% of the dataset).",
            "evidence": {"duplicate_count": dup_count, "duplicate_percentage": round(dup_count/total_rows*100, 2)},
            "takeaway": f"{dup_count} duplicate row(s) identified." if dup_count > 0 else "Zero duplicate rows found."
        }

    # -------------------------------------------------------------------------
    # 2. Dynamic Grouping / Breakdown Query (e.g. "revenue by hotel", "runs by player", "X per Y")
    # -------------------------------------------------------------------------
    is_grouped = any(k in q_lower for k in ["by", "per", "for each", "each", "wise", "breakdown", "grouped by"]) or (len(cat_matches) > 0 and len(num_matches) > 0)

    if is_grouped and cat_matches:
        group_col = cat_matches[0]
        
        # Determine aggregation
        agg = "sum"
        agg_title = "Total"
        if any(w in q_lower for w in ["average", "avg", "mean"]):
            agg = "mean"
            agg_title = "Average"
        elif any(w in q_lower for w in ["count", "number of", "frequency", "how many"]):
            agg = "count"
            agg_title = "Count"
        elif any(w in q_lower for w in ["max", "maximum", "highest", "peak"]):
            agg = "max"
            agg_title = "Maximum"
        elif any(w in q_lower for w in ["min", "minimum", "lowest"]):
            agg = "min"
            agg_title = "Minimum"

        # Check for top N / bottom N
        top_n = None
        top_match = re.search(r'(?:top|first|highest)\s*(\d+)', q_lower)
        if top_match:
            top_n = int(top_match.group(1))
        bottom_match = re.search(r'(?:bottom|lowest|last)\s*(\d+)', q_lower)
        bottom_n = int(bottom_match.group(1)) if bottom_match else None

        if num_matches:
            val_col = num_matches[0]
            grouped = df.groupby(group_col)[val_col].agg(agg).dropna()
            if bottom_n:
                grouped = grouped.sort_values(ascending=True).head(bottom_n)
            else:
                grouped = grouped.sort_values(ascending=False)
                if top_n:
                    grouped = grouped.head(top_n)

            lines = []
            for k, v in grouped.items():
                v_str = f"{v:,.2f}" if isinstance(v, (float, np.floating)) else f"{v:,}"
                lines.append(f"• {k}: {v_str}")

            total_val = float(df[val_col].sum()) if agg == "sum" and pd.api.types.is_numeric_dtype(df[val_col]) else None
            top_item = str(grouped.index[0]) if not grouped.empty else "N/A"
            top_val = float(grouped.iloc[0]) if not grouped.empty else 0.0

            brief_analysis = ""
            if len(grouped) > 1 and top_val > 0 and total_val and total_val > 0:
                share_pct = (top_val / total_val) * 100
                brief_analysis = f"Brief analysis: '{top_item}' leads with {top_val:,.2f} ({share_pct:.1f}% of total {val_col})."

            limit_display = lines[:40]
            remaining = len(lines) - len(limit_display)
            res_text = f"{agg_title} '{val_col}' by '{group_col}' ({len(grouped)} groups):\n" + "\n".join(limit_display)
            if remaining > 0:
                res_text += f"\n... and {remaining} additional group(s)."
            if brief_analysis:
                res_text += f"\n\n{brief_analysis.strip()}"

            return {
                "question": question,
                "answer": res_text,
                "evidence": {
                    "group_by": str(group_col),
                    "metric": str(val_col),
                    "aggregation": agg,
                    "results_count": len(grouped),
                    "breakdown": {str(k): round(float(v), 2) if isinstance(v, (float, np.floating, int, np.integer)) else str(v) for k, v in grouped.head(30).items()}
                },
                "takeaway": f"Highest in '{group_col}' is '{top_item}' with {top_val:,.2f} {val_col}."
            }
        else:
            # Categorical count grouping
            counts = df[group_col].value_counts(dropna=True)
            if top_n:
                counts = counts.head(top_n)
            elif bottom_n:
                counts = counts.tail(bottom_n)

            lines = [f"• {k}: {v:,} record(s)" for k, v in counts.items()]
            top_item = str(counts.index[0]) if not counts.empty else "N/A"
            top_cnt = int(counts.iloc[0]) if not counts.empty else 0

            return {
                "question": question,
                "answer": f"Record count by '{group_col}' ({len(counts)} distinct groups):\n" + "\n".join(lines[:35]),
                "evidence": {
                    "group_by": str(group_col),
                    "counts": {str(k): int(v) for k, v in counts.head(30).items()}
                },
                "takeaway": f"Most frequent '{group_col}' is '{top_item}' with {top_cnt:,} records ({top_cnt/total_rows*100:.1f}% of dataset)."
            }

    # -------------------------------------------------------------------------
    # 3. List / Distinct Values Query (e.g. "list all categories", "what are the teams")
    # -------------------------------------------------------------------------
    if any(k in q_lower for k in ["list", "what are the", "distinct", "unique", "show all", "names of", "all "]) and col_matches:
        target_col = col_matches[0][0]
        uniques = df[target_col].dropna().unique().tolist()
        formatted_uniques = [str(u) for u in uniques[:50]]
        rem = len(uniques) - len(formatted_uniques)
        res_str = f"Distinct values for '{target_col}' ({len(uniques):,} total):\n" + "\n".join([f"• {u}" for u in formatted_uniques])
        if rem > 0:
            res_str += f"\n... and {rem:,} more unique values."

        return {
            "question": question,
            "answer": res_str,
            "evidence": {
                "column": str(target_col),
                "unique_count": len(uniques),
                "sample_values": formatted_uniques[:20]
            },
            "takeaway": f"Found {len(uniques):,} distinct values for '{target_col}'."
        }

    # -------------------------------------------------------------------------
    # 4. Extremes / Top / Bottom / Ranking Query on Single Column
    # -------------------------------------------------------------------------
    if any(k in q_lower for k in ["highest", "maximum", "max", "top", "lowest", "minimum", "min", "bottom", "best", "worst"]):
        target_col = num_matches[0] if num_matches else (col_matches[0][0] if col_matches else None)
        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            clean_s = df[target_col].dropna()
            is_highest = any(k in q_lower for k in ["highest", "maximum", "max", "top", "best"])
            val = float(clean_s.max()) if is_highest else float(clean_s.min())
            label = "Maximum (highest)" if is_highest else "Minimum (lowest)"

            matching_rows = df[df[target_col] == val]
            context_cols = [c for c in df.columns if c != target_col][:3]
            row_details = []
            if not matching_rows.empty and context_cols:
                row_dict = matching_rows.iloc[0][context_cols].to_dict()
                row_details = [f"{k}: {v}" for k, v in row_dict.items()]

            row_str = f" Context: ({', '.join(row_details)})." if row_details else ""
            return {
                "question": question,
                "answer": f"The {label} value for '{target_col}' in the dataset is {val:,.2f}.{row_str}",
                "evidence": {
                    "column": str(target_col),
                    "metric": "max" if is_highest else "min",
                    "value": val,
                    "mean_comparison": round(float(clean_s.mean()), 2),
                    "median_comparison": round(float(clean_s.median()), 2)
                },
                "takeaway": f"{label} '{target_col}' is {val:,.2f} (dataset average is {clean_s.mean():,.2f})."
            }

    # -------------------------------------------------------------------------
    # 5. Aggregates / Average / Total Query on Single Column
    # -------------------------------------------------------------------------
    if num_matches:
        target_num = num_matches[0]
        series = df[target_num].dropna()
        if not series.empty:
            if any(k in q_lower for k in ["total", "sum", "aggregate"]):
                tot = float(series.sum())
                return {
                    "question": question,
                    "answer": f"The total sum for '{target_num}' across all {len(series):,} valid rows is {tot:,.2f}.",
                    "evidence": {"column": str(target_num), "sum": tot, "count": len(series)},
                    "takeaway": f"Total {target_num}: {tot:,.2f}."
                }
            elif any(k in q_lower for k in ["average", "mean", "avg"]):
                avg = float(series.mean())
                med = float(series.median())
                return {
                    "question": question,
                    "answer": f"The mean average for '{target_num}' is {avg:,.2f} (median: {med:,.2f}, standard deviation: {series.std():,.2f}).",
                    "evidence": {"column": str(target_num), "mean": avg, "median": med, "std": round(float(series.std()), 2)},
                    "takeaway": f"Average {target_num}: {avg:,.2f}."
                }
            elif any(k in q_lower for k in ["median", "middle"]):
                med = float(series.median())
                return {
                    "question": question,
                    "answer": f"The median value for '{target_num}' is {med:,.2f}.",
                    "evidence": {"column": str(target_num), "median": med},
                    "takeaway": f"Median {target_num}: {med:,.2f}."
                }

    # -------------------------------------------------------------------------
    # 6. Single Column Profile / Drilldown Query
    # -------------------------------------------------------------------------
    if col_matches:
        target_col = col_matches[0][0]
        series = df[target_col]
        null_count = int(series.isnull().sum())
        uniques_cnt = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            clean_s = series.dropna()
            return {
                "question": question,
                "answer": (
                    f"Statistical summary for '{target_col}':\n"
                    f"• Mean: {clean_s.mean():,.2f}\n"
                    f"• Median: {clean_s.median():,.2f}\n"
                    f"• Min: {clean_s.min():,.2f}\n"
                    f"• Max: {clean_s.max():,.2f}\n"
                    f"• Missing: {null_count:,} ({null_count/total_rows*100:.1f}%)"
                ),
                "evidence": {
                    "column": str(target_col),
                    "mean": round(float(clean_s.mean()), 2),
                    "median": round(float(clean_s.median()), 2),
                    "min": round(float(clean_s.min()), 2),
                    "max": round(float(clean_s.max()), 2),
                    "null_count": null_count
                },
                "takeaway": f"'{target_col}' ranges from {clean_s.min():,.2f} to {clean_s.max():,.2f}."
            }
        else:
            top_vals = series.value_counts(dropna=True).head(5).to_dict()
            val_lines = [f"• {k}: {v:,} occurrences ({v/total_rows*100:.1f}%)" for k, v in top_vals.items()]
            return {
                "question": question,
                "answer": f"Summary for '{target_col}' ({uniques_cnt:,} unique values, {null_count:,} missing):\n" + "\n".join(val_lines),
                "evidence": {
                    "column": str(target_col),
                    "unique_values": uniques_cnt,
                    "top_values": {str(k): int(v) for k, v in top_vals.items()}
                },
                "takeaway": f"Most common '{target_col}' is '{series.mode().iloc[0] if not series.empty else 'N/A'}'."
            }

    # -------------------------------------------------------------------------
    # 7. Correlation Query
    # -------------------------------------------------------------------------
    if any(k in q_lower for k in ["correlat", "relationship", "relation", "linked", "associate"]):
        num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols_all) >= 2:
            corr_matrix = df[num_cols_all].corr().fillna(0)
            pairs = []
            for i in range(len(num_cols_all)):
                for j in range(i + 1, len(num_cols_all)):
                    c1, c2 = num_cols_all[i], num_cols_all[j]
                    r_val = float(corr_matrix.loc[c1, c2])
                    pairs.append((c1, c2, r_val))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if pairs:
                top = pairs[0]
                lines = [f"• '{p[0]}' & '{p[1]}': r = {p[2]:.3f}" for p in pairs[:5]]
                return {
                    "question": question,
                    "answer": f"Top correlations calculated from the uploaded dataset:\n" + "\n".join(lines),
                    "evidence": {"top_correlations": [{"feature_1": p[0], "feature_2": p[1], "correlation": round(p[2], 3)} for p in pairs[:5]]},
                    "takeaway": f"Strongest relationship is between '{top[0]}' and '{top[1]}' (r = {top[2]:.3f})."
                }

    # -------------------------------------------------------------------------
    # 8. Unanswerable Question / General Fallback: Clear notification
    # -------------------------------------------------------------------------
    cols_preview = [str(c) for c in df.columns[:8]]
    num_cols_count = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])])
    cat_cols_count = total_cols - num_cols_count

    return {
        "question": question,
        "answer": (
            f"The required information to answer this question is not available in the uploaded dataset.\n\n"
            f"The dataset contains {total_rows:,} records and {total_cols} columns: "
            f"{', '.join(cols_preview)}{'...' if total_cols > 8 else ''}.\n"
            f"Please ask a question based on the available columns and records."
        ),
        "evidence": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": [str(c) for c in df.columns]
        },
        "takeaway": "Required information is not available in this dataset."
    }

