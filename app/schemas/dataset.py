from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Dataset Summary
class ColumnSummary(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    data_type_issue: Optional[Dict[str, Any]] = None

class OutlierSummary(BaseModel):
    column: str
    outlier_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float
    method: str
    sample_outliers: List[float]

class InconsistencySummary(BaseModel):
    column: str
    issue_type: str
    description: str
    affected_count: int
    severity: str  # high, medium, low

class QualityAnalysisResponse(BaseModel):
    overall_quality_score: float  # 0 to 100
    completeness_score: float
    uniqueness_score: float
    consistency_score: float
    validity_score: float
    
    total_rows: int
    total_columns: int
    total_cells: int
    total_missing_cells: int
    missing_cells_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    
    columns_summary: List[ColumnSummary]
    outliers: List[OutlierSummary]
    inconsistencies: List[InconsistencySummary]
    correlations: Dict[str, Dict[str, float]]
    important_variables: List[Dict[str, Any]]
    trends_and_patterns: List[Dict[str, Any]]
    next_investigations: List[str]
    trim_spaces_issues: Optional[Dict[str, Any]] = None

# Cleaning Plan & Decisions
class ColumnCleaningDecision(BaseModel):
    column: str
    action: str  # 'fill', 'remove_column', 'remove_rows', 'keep'
    fill_method: Optional[str] = None  # 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill'
    fill_value: Optional[Any] = None
    outlier_action: Optional[str] = "keep"  # 'clip', 'remove_rows', 'keep'
    convert_type: Optional[str] = None  # 'integer', 'decimal', 'datetime', 'boolean', 'string'
    rationale: Optional[str] = None

class CleaningPlanRequest(BaseModel):
    decisions: List[ColumnCleaningDecision]
    remove_duplicates: bool = False
    trim_spaces: bool = False
    custom_notes: Optional[str] = None

class AuditLogItem(BaseModel):
    id: int
    step_number: int
    action_type: str
    target_column: Optional[str]
    decision_details: str
    rationale: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    created_at: str

class CleaningResultResponse(BaseModel):
    dataset_id: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    missing_cells_before: int
    missing_cells_after: int
    duplicates_before: int
    duplicates_after: int
    quality_score_before: float
    quality_score_after: float
    audit_logs: List[AuditLogItem]
    sample_preview: List[Dict[str, Any]]

# Visualizations
class ChartDataRequest(BaseModel):
    chart_type: str  # 'bar', 'scatter', 'line', 'box', 'histogram', 'heatmap', 'pie'
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    aggregation: Optional[str] = "none"  # 'none', 'sum', 'mean', 'count', 'median'
    title: Optional[str] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    file_type: str
    row_count: int
    column_count: int
    status: str
    created_at: str
    updated_at: str
    file_size_bytes: int
    quality_score: Optional[float] = None
    cleaned: bool = False

# User Questions & Dataset Q&A
class UserQuestionRequest(BaseModel):
    question: str

class UserQuestionResponse(BaseModel):
    question: str
    answer: str
    evidence: Dict[str, Any]
    takeaway: str
