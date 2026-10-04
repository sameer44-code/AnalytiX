import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

BLUE_GRAY_PALETTE = ["#2563eb", "#475569", "#0f766e", "#64748b", "#334155", "#0284c7", "#d97706", "#4f46e5"]

def get_blue_gray_layout(title: str, x_label: str = "", y_label: str = "") -> Dict[str, Any]:
    return {
        "title": {
            "text": title,
            "font": {"family": "Charter, Sitka Text, Cambria, Georgia, serif", "size": 15, "color": "#1e293b"}
        },
        "margin": {"l": 55, "r": 25, "t": 55, "b": 55},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f8fafc",
        "xaxis": {
            "title": x_label,
            "gridcolor": "#e2e8f0",
            "zerolinecolor": "#cbd5e1",
            "tickfont": {"family": "Charter, Cambria, Georgia, serif", "size": 11, "color": "#475569"},
            "automargin": True
        },
        "yaxis": {
            "title": y_label,
            "gridcolor": "#e2e8f0",
            "zerolinecolor": "#cbd5e1",
            "tickfont": {"family": "Charter, Cambria, Georgia, serif", "size": 11, "color": "#475569"},
            "automargin": True
        },
        "hoverlabel": {
            "bgcolor": "#1e293b",
            "font": {"family": "Charter, Cambria, Georgia, serif", "size": 11, "color": "#ffffff"}
        },
        "showlegend": True,
        "legend": {"font": {"family": "Charter, Cambria, Georgia, serif", "size": 10, "color": "#475569"}}
    }


def _pick_best_numeric_cols(df: pd.DataFrame, n: int = 4) -> List[str]:
    """Returns numeric columns with the highest variance (most informative)."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    if not num_cols:
        return []
    variances = {c: float(df[c].dropna().var()) for c in num_cols}
    return sorted(variances, key=lambda c: variances[c], reverse=True)[:n]


def _pick_best_categorical_col(df: pd.DataFrame, max_categories: int = 30) -> Optional[str]:
    """Returns the best categorical column for grouping: not too many, not too few categories."""
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])
                and 2 <= df[c].nunique(dropna=True) <= max_categories]
    if not cat_cols:
        return None
    # Prefer columns with moderate cardinality (better for grouping)
    return sorted(cat_cols, key=lambda c: abs(df[c].nunique(dropna=True) - 10))[0]


def _pick_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """Returns a datetime column if available, or a column with 'date' or 'time' in its name."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        return dt_cols[0]
    # Try columns that look like dates by name
    name_hints = [c for c in df.columns if any(k in str(c).lower() for k in ['date', 'time', 'year', 'month', 'day'])]
    return name_hints[0] if name_hints else None


def generate_recommended_visualizations(
    df: pd.DataFrame,
    quality_analysis: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Generates up to 6 meaningful visualizations from the actual dataset.
    Each chart must communicate a real insight: distribution, comparison, relationship, or trend.
    Charts are selected based on available column types and data quality.
    """
    charts = []
    num_cols = _pick_best_numeric_cols(df, n=6)
    cat_col = _pick_best_categorical_col(df)
    dt_col = _pick_datetime_col(df)

    all_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    all_cat = [c for c in df.columns
               if not pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_datetime64_any_dtype(df[c])
               and df[c].nunique(dropna=True) >= 2]

    # ─────────────────────────────────────────────────────────────
    # CHART 1: Categorical Bar Chart (best categorical vs best numeric)
    # Insight: Which category has the highest/lowest total or average of a key metric?
    # ─────────────────────────────────────────────────────────────
    if cat_col and num_cols:
        metric_col = num_cols[0]
        agg = df.groupby(cat_col)[metric_col].sum().dropna().sort_values(ascending=False).head(15)
        if len(agg) >= 2:
            charts.append({
                "id": "rec_chart_bar",
                "title": f"Total {metric_col} by {cat_col}",
                "description": f"Which '{cat_col}' contributes most to total '{metric_col}'?",
                "type": "bar",
                "spec": {
                    "data": [{
                        "x": [str(x) for x in agg.index.tolist()],
                        "y": [round(float(y), 2) for y in agg.values.tolist()],
                        "type": "bar",
                        "marker": {
                            "color": [BLUE_GRAY_PALETTE[i % len(BLUE_GRAY_PALETTE)] for i in range(len(agg))],
                            "line": {"color": "#1d4ed8", "width": 0.5}
                        },
                        "name": f"Sum of {metric_col}"
                    }],
                    "layout": get_blue_gray_layout(
                        f"Total {metric_col} by {cat_col}",
                        x_label=cat_col,
                        y_label=f"Sum of {metric_col}"
                    )
                }
            })

    # ─────────────────────────────────────────────────────────────
    # CHART 2: Distribution Histogram (most variable numeric column)
    # Insight: How is the key metric distributed across records?
    # ─────────────────────────────────────────────────────────────
    if num_cols:
        dist_col = num_cols[0]
        series = df[dist_col].dropna()
        if len(series) >= 5:
            sample = series.sample(min(len(series), 2000), random_state=42).tolist()
            charts.append({
                "id": "rec_chart_dist",
                "title": f"Distribution of {dist_col}",
                "description": f"How are values of '{dist_col}' distributed across all records?",
                "type": "histogram",
                "spec": {
                    "data": [{
                        "x": sample,
                        "type": "histogram",
                        "marker": {"color": "#475569", "opacity": 0.85},
                        "nbinsx": min(30, max(10, series.nunique() // 3)),
                        "name": dist_col
                    }],
                    "layout": get_blue_gray_layout(
                        f"Distribution: {dist_col}",
                        x_label=dist_col,
                        y_label="Count of Records"
                    )
                }
            })

    # ─────────────────────────────────────────────────────────────
    # CHART 3: Correlation Heatmap (only when 3+ numeric cols with meaningful correlations)
    # Insight: Which numeric features move together?
    # ─────────────────────────────────────────────────────────────
    heatmap_cols = [c for c in all_num if df[c].nunique() > 5][:8]
    if len(heatmap_cols) >= 3:
        corr_matrix = df[heatmap_cols].corr().fillna(0).round(2)
        # Check if there's at least one meaningful correlation
        off_diag = [abs(corr_matrix.iloc[i, j])
                    for i in range(len(heatmap_cols))
                    for j in range(len(heatmap_cols)) if i != j]
        max_corr = max(off_diag) if off_diag else 0
        if max_corr >= 0.2:
            col_names = [str(c) for c in corr_matrix.columns]
            charts.append({
                "id": "rec_chart_corr",
                "title": "Feature Correlation Heatmap",
                "description": "Pairwise linear correlation between numeric features. Dark = stronger relationship.",
                "type": "heatmap",
                "spec": {
                    "data": [{
                        "z": corr_matrix.values.tolist(),
                        "x": col_names,
                        "y": col_names,
                        "type": "heatmap",
                        "colorscale": [[0, "#f8fafc"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
                        "hoverongaps": False,
                        "colorbar": {"title": "r"}
                    }],
                    "layout": get_blue_gray_layout("Feature Correlation Heatmap")
                }
            })

    # ─────────────────────────────────────────────────────────────
    # CHART 4: Scatter Plot (two most correlated or most variable numeric columns)
    # Insight: Is there a relationship between two key metrics?
    # ─────────────────────────────────────────────────────────────
    if len(num_cols) >= 2:
        # Prefer the most correlated pair
        x_c, y_c = num_cols[0], num_cols[1]
        if len(heatmap_cols) >= 2:
            try:
                corr_mat = df[heatmap_cols].corr().fillna(0)
                best_pair = (None, None, 0.0)
                for i in range(len(heatmap_cols)):
                    for j in range(i + 1, len(heatmap_cols)):
                        c1, c2 = heatmap_cols[i], heatmap_cols[j]
                        r = abs(float(corr_mat.loc[c1, c2]))
                        if r > best_pair[2]:
                            best_pair = (c1, c2, r)
                if best_pair[0] and best_pair[2] >= 0.1:
                    x_c, y_c = best_pair[0], best_pair[1]
            except Exception:
                pass

        valid_pairs = df[[x_c, y_c]].dropna()
        if len(valid_pairs) >= 5:
            sample_df = valid_pairs.sample(min(len(valid_pairs), 800), random_state=42)
            # Color by category if available
            color_data = {}
            if cat_col and len(df[cat_col].dropna().unique()) <= 15:
                cat_series = df.loc[valid_pairs.index, cat_col].fillna("Unknown")
                cat_sample = cat_series.sample(min(len(cat_series), 800), random_state=42) if len(cat_series) > 800 else cat_series
                unique_cats = cat_sample.unique().tolist()
                color_map = {c: BLUE_GRAY_PALETTE[i % len(BLUE_GRAY_PALETTE)] for i, c in enumerate(unique_cats)}
                sample_idx = sample_df.index
                colors = [color_map.get(str(df.at[idx, cat_col] if pd.notna(df.at[idx, cat_col]) else "Unknown"), "#475569") for idx in sample_idx]
                color_data = {"marker": {"color": colors, "size": 6, "opacity": 0.75}}
            else:
                color_data = {"marker": {"color": "#0f766e", "size": 6, "opacity": 0.75}}

            charts.append({
                "id": "rec_chart_scatter",
                "title": f"{x_c} vs {y_c}",
                "description": f"Does '{x_c}' correlate with '{y_c}'? Each point is one record.",
                "type": "scatter",
                "spec": {
                    "data": [{
                        "x": sample_df[x_c].tolist(),
                        "y": sample_df[y_c].tolist(),
                        "mode": "markers",
                        "type": "scatter",
                        "name": "Records",
                        **color_data
                    }],
                    "layout": get_blue_gray_layout(
                        f"{x_c} vs {y_c}",
                        x_label=str(x_c),
                        y_label=str(y_c)
                    )
                }
            })

    # ─────────────────────────────────────────────────────────────
    # CHART 5: Box Plot (all key numeric cols — spread & outlier comparison)
    # Insight: Which columns have high variance or extreme outliers?
    # ─────────────────────────────────────────────────────────────
    box_cols = num_cols[:4]
    if box_cols:
        # Normalize each column to compare on same scale (use z-score or just show side by side)
        traces = []
        for i, bc in enumerate(box_cols):
            clean_box = df[bc].dropna()
            if len(clean_box) < 5:
                continue
            sample_vals = clean_box.sample(min(len(clean_box), 800), random_state=42).tolist()
            traces.append({
                "y": sample_vals,
                "type": "box",
                "name": str(bc),
                "marker": {"color": BLUE_GRAY_PALETTE[i % len(BLUE_GRAY_PALETTE)]},
                "boxpoints": "outliers"
            })

        if traces:
            col_names_str = ", ".join(str(c) for c in box_cols)
            charts.append({
                "id": "rec_chart_box",
                "title": "Spread & Outlier Comparison",
                "description": f"Quartile distribution and outliers for key numeric columns: {col_names_str}.",
                "type": "box",
                "spec": {
                    "data": traces,
                    "layout": get_blue_gray_layout("Quartile Spread & Outliers")
                }
            })

    # ─────────────────────────────────────────────────────────────
    # CHART 6: Time-Series Line OR Category Distribution Pie
    # Line: best when datetime column available with numeric metric
    # Pie: best when a categorical column represents meaningful proportions
    # ─────────────────────────────────────────────────────────────
    if dt_col and num_cols and len(charts) < 6:
        try:
            time_series = df[[dt_col, num_cols[0]]].dropna().copy()
            if not pd.api.types.is_datetime64_any_dtype(time_series[dt_col]):
                time_series[dt_col] = pd.to_datetime(time_series[dt_col], errors='coerce')
            time_series = time_series.dropna().sort_values(dt_col)
            if len(time_series) >= 5:
                if len(time_series) > 500:
                    # Aggregate by month or quarter to reduce points
                    time_series = time_series.set_index(dt_col).resample('ME')[num_cols[0]].sum().reset_index()
                charts.append({
                    "id": "rec_chart_line",
                    "title": f"{num_cols[0]} Over Time",
                    "description": f"How does '{num_cols[0]}' change over time ({dt_col})?",
                    "type": "line",
                    "spec": {
                        "data": [{
                            "x": [str(x) for x in time_series[dt_col].tolist()],
                            "y": [round(float(y), 2) for y in time_series[num_cols[0]].tolist()],
                            "type": "scatter",
                            "mode": "lines+markers",
                            "line": {"color": "#2563eb", "width": 2},
                            "marker": {"size": 4, "color": "#1d4ed8"},
                            "name": str(num_cols[0])
                        }],
                        "layout": get_blue_gray_layout(
                            f"{num_cols[0]} Over Time",
                            x_label=str(dt_col),
                            y_label=str(num_cols[0])
                        )
                    }
                })
        except Exception:
            pass

    # Pie fallback (category share) when we still have room and no time chart added
    if len(charts) < 6 and cat_col and num_cols:
        pie_agg = df.groupby(cat_col)[num_cols[0]].sum().dropna().sort_values(ascending=False).head(10)
        if len(pie_agg) >= 3:
            charts.append({
                "id": "rec_chart_pie",
                "title": f"Share of {num_cols[0]} by {cat_col}",
                "description": f"Proportional contribution of each '{cat_col}' to total '{num_cols[0]}'.",
                "type": "pie",
                "spec": {
                    "data": [{
                        "labels": [str(x) for x in pie_agg.index.tolist()],
                        "values": [round(float(v), 2) for v in pie_agg.values.tolist()],
                        "type": "pie",
                        "hole": 0.4,
                        "marker": {
                            "colors": BLUE_GRAY_PALETTE[:len(pie_agg)]
                        }
                    }],
                    "layout": get_blue_gray_layout(f"Share of {num_cols[0]} by {cat_col}")
                }
            })
    elif len(charts) < 6 and all_cat and not cat_col:
        # Fallback: value counts for the first categorical column if no good grouping column found
        fc = all_cat[0]
        counts = df[fc].value_counts().head(15)
        if len(counts) >= 2:
            charts.append({
                "id": "rec_chart_cat_bar",
                "title": f"Frequency of {fc}",
                "description": f"How many records belong to each '{fc}' category?",
                "type": "bar",
                "spec": {
                    "data": [{
                        "x": [str(x) for x in counts.index.tolist()],
                        "y": [int(y) for y in counts.values.tolist()],
                        "type": "bar",
                        "marker": {"color": "#2563eb"},
                        "name": f"Count of {fc}"
                    }],
                    "layout": get_blue_gray_layout(
                        f"Category Frequency: {fc}",
                        x_label=str(fc),
                        y_label="Record Count"
                    )
                }
            })

    return charts[:6]


def generate_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str = None,
    y_col: str = None,
    color_col: str = None,
    aggregation: str = "none",
    title: str = None
) -> Dict[str, Any]:
    """
    Executes a custom chart specification with performance optimization and blue-gray styling.
    """
    clean_df = df.copy()
    req_title = title or f"{chart_type.upper()} Visual Query"

    if chart_type == "bar":
        if not x_col:
            x_col = clean_df.columns[0]

        if y_col and aggregation != "none":
            if aggregation == "sum":
                grouped = clean_df.groupby(x_col)[y_col].sum()
            elif aggregation == "count":
                grouped = clean_df.groupby(x_col)[y_col].count()
            elif aggregation == "median":
                grouped = clean_df.groupby(x_col)[y_col].median()
            else:
                grouped = clean_df.groupby(x_col)[y_col].mean()

            grouped = grouped.dropna().sort_values(ascending=False).head(20)
            x_data = [str(x) for x in grouped.index]
            y_data = [round(float(y), 2) for y in grouped.values]
            y_title = f"{aggregation.capitalize()} of {y_col}"
        elif y_col:
            sub = clean_df[[x_col, y_col]].dropna().head(30)
            x_data = [str(x) for x in sub[x_col]]
            y_data = pd.to_numeric(sub[y_col], errors="coerce").fillna(0).tolist()
            y_title = y_col
        else:
            counts = clean_df[x_col].value_counts().head(20)
            x_data = [str(x) for x in counts.index]
            y_data = [int(y) for y in counts.values]
            y_title = "Record Count"

        trace = {
            "x": x_data,
            "y": y_data,
            "type": "bar",
            "marker": {"color": "#2563eb"}
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, x_label=x_col or "", y_label=y_title)
        }

    elif chart_type == "line":
        sub = clean_df.dropna(subset=[c for c in [x_col, y_col] if c])
        if len(sub) > 800:
            sub = sub.head(800)

        x_data = sub[x_col].tolist() if x_col else list(range(len(sub)))
        y_data = pd.to_numeric(sub[y_col], errors="coerce").fillna(0).tolist() if y_col else list(range(len(sub)))

        trace = {
            "x": [str(x) for x in x_data],
            "y": y_data,
            "type": "scatter",
            "mode": "lines+markers",
            "line": {"color": "#2563eb", "width": 2},
            "marker": {"size": 4, "color": "#1d4ed8"}
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, x_label=x_col or "Index", y_label=y_col or "")
        }

    elif chart_type == "scatter":
        sub = clean_df.dropna(subset=[c for c in [x_col, y_col] if c])
        if len(sub) > 1000:
            sub = sub.sample(1000, random_state=42)

        trace = {
            "x": sub[x_col].tolist() if x_col else [],
            "y": sub[y_col].tolist() if y_col else [],
            "type": "scatter",
            "mode": "markers",
            "marker": {"color": "#0f766e", "size": 6, "opacity": 0.75}
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, x_label=x_col or "", y_label=y_col or "")
        }

    elif chart_type == "histogram":
        hist_col = x_col or y_col
        if not hist_col:
            num_c = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])]
            hist_col = num_c[0] if num_c else clean_df.columns[0] if len(clean_df.columns) > 0 else None
        if hist_col is None:
            return {"data": [], "layout": get_blue_gray_layout(req_title)}
        series = pd.to_numeric(clean_df[hist_col], errors="coerce").dropna()
        if len(series) > 1500:
            series = series.sample(1500, random_state=42)

        trace = {
            "x": series.tolist(),
            "type": "histogram",
            "marker": {"color": "#475569"}
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, x_label=hist_col, y_label="Count")
        }

    elif chart_type == "box":
        target = y_col or x_col
        if not target:
            num_c = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])]
            target = num_c[0] if num_c else clean_df.columns[0] if len(clean_df.columns) > 0 else None
        if target is None:
            return {"data": [], "layout": get_blue_gray_layout(req_title)}
        series = pd.to_numeric(clean_df[target], errors="coerce").dropna()
        if len(series) > 1200:
            series = series.sample(1200, random_state=42)

        trace = {
            "y": series.tolist(),
            "type": "box",
            "name": target,
            "marker": {"color": "#64748b"}
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, y_label=target)
        }

    elif chart_type == "area":
        sub = clean_df.dropna(subset=[c for c in [x_col, y_col] if c])
        if len(sub) > 800:
            sub = sub.head(800)
        x_data = sub[x_col].tolist() if x_col else list(range(len(sub)))
        y_data = pd.to_numeric(sub[y_col], errors="coerce").fillna(0).tolist() if y_col else list(range(len(sub)))
        trace = {
            "x": [str(x) for x in x_data],
            "y": y_data,
            "type": "scatter",
            "mode": "lines",
            "fill": "tozeroy",
            "line": {"color": "#2563eb", "width": 2},
            "fillcolor": "rgba(37, 99, 235, 0.2)"
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title, x_label=x_col or "Index", y_label=y_col or "")
        }

    elif chart_type == "pie":
        target_cat = x_col or y_col
        if not target_cat:
            cat_candidates = [c for c in clean_df.columns if not pd.api.types.is_numeric_dtype(clean_df[c])]
            target_cat = cat_candidates[0] if cat_candidates else clean_df.columns[0]

        if y_col and y_col != target_cat and pd.api.types.is_numeric_dtype(clean_df[y_col]):
            counts = clean_df.groupby(target_cat)[y_col].sum().sort_values(ascending=False).head(10)
        else:
            counts = clean_df[target_cat].value_counts().head(10)

        trace = {
            "labels": [str(x) for x in counts.index],
            "values": [float(v) for v in counts.values],
            "type": "pie",
            "hole": 0.45,
            "marker": {
                "colors": ["#1e293b", "#2563eb", "#0f766e", "#d97706", "#64748b", "#4f46e5", "#0284c7", "#e11d48", "#84cc16", "#a855f7"]
            }
        }
        return {
            "data": [trace],
            "layout": get_blue_gray_layout(req_title)
        }

    return {
        "data": [],
        "layout": get_blue_gray_layout(req_title)
    }
