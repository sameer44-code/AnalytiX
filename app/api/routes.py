import os
import uuid
import json
import datetime
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings, UPLOAD_DIR, CLEANED_DIR, REPORTS_DIR
from app.core.database import get_db
from app.models.dataset import Dataset, AuditLog
from app.schemas.dataset import (
    CleaningPlanRequest, CleaningResultResponse, ChartDataRequest, DatasetResponse,
    UserQuestionRequest, UserQuestionResponse
)
from app.services.file_parser import parse_uploaded_file
from app.services.sample_data import SAMPLE_CATALOG
from app.services.data_analyzer import analyze_dataset
from app.services.data_cleaner import execute_cleaning_plan
from app.services.question_answering import answer_dataset_question
from app.services.visualization_service import (
    generate_recommended_visualizations, generate_custom_chart
)
from app.services.pdf_report_service import generate_pdf_report

router = APIRouter()

# In-memory performance cache for parsed dataframes
_DF_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}

def get_current_df(dataset: Dataset) -> pd.DataFrame:
    """Returns the cleaned dataframe if available, otherwise raw dataframe, with caching."""
    target_path = dataset.cleaned_file_path if dataset.cleaned_file_path and os.path.exists(dataset.cleaned_file_path) else dataset.file_path
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Underlying dataset file not found on server.")
    
    mtime = os.path.getmtime(target_path)
    if target_path in _DF_CACHE:
        cached_mtime, cached_df = _DF_CACHE[target_path]
        if cached_mtime == mtime:
            return cached_df
    
    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(target_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(target_path)
    else:
        df, _ = parse_uploaded_file(target_path, os.path.basename(target_path))
    
    _DF_CACHE[target_path] = (mtime, df)
    return df

@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    results = []
    for d in datasets:
        q = d.get_quality_analysis()
        results.append(DatasetResponse(
            id=d.id,
            name=d.name,
            file_type=d.file_type,
            row_count=d.row_count,
            column_count=d.column_count,
            status=d.status,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat(),
            file_size_bytes=d.file_size_bytes,
            quality_score=q.get("overall_quality_score"),
            cleaned=bool(d.cleaned_file_path)
        ))
    return results

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Uploads and ingests any supported file (CSV, XLSX, HTML, PDF, DOT)."""
    dataset_id = str(uuid.uuid4())
    filename = file.filename or "uploaded_data.csv"
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{filename}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        df, detected_type = parse_uploaded_file(save_path, filename)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {str(e)}")

    # Initial analysis
    quality = analyze_dataset(df)

    # Save to database
    dataset = Dataset(
        id=dataset_id,
        name=filename,
        file_type=detected_type,
        file_path=save_path,
        file_size_bytes=len(content),
        row_count=len(df),
        column_count=len(df.columns),
        status="analyzed",
        metadata_json=json.dumps({
            "columns": list(df.columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
        }),
        quality_analysis_json=json.dumps(quality)
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_type": dataset.file_type,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "quality_score": quality.get("overall_quality_score"),
        "message": f"Successfully ingested {filename} with {len(df)} rows and {len(df.columns)} columns."
    }

@router.post("/sample/{sample_key}")
def load_sample_dataset(sample_key: str, db: Session = Depends(get_db)):
    """Instant loads pre-seeded enterprise sample datasets."""
    if sample_key not in SAMPLE_CATALOG:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_key}' not found. Available: {list(SAMPLE_CATALOG.keys())}")

    meta = SAMPLE_CATALOG[sample_key]
    df = meta["generator"]()
    dataset_id = str(uuid.uuid4())
    save_filename = f"{sample_key}.csv"
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{save_filename}")
    df.to_csv(save_path, index=False)

    quality = analyze_dataset(df)

    dataset = Dataset(
        id=dataset_id,
        name=meta["name"],
        file_type="csv",
        file_path=save_path,
        file_size_bytes=os.path.getsize(save_path),
        row_count=len(df),
        column_count=len(df.columns),
        status="analyzed",
        metadata_json=json.dumps({
            "columns": list(df.columns),
            "description": meta["description"],
            "sample_key": sample_key
        }),
        quality_analysis_json=json.dumps(quality)
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_type": dataset.file_type,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "quality_score": quality.get("overall_quality_score"),
        "message": f"Sample dataset '{meta['name']}' ready for analysis."
    }

@router.get("/datasets/{dataset_id}")
def get_dataset_details(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_type": dataset.file_type,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat(),
        "cleaned": bool(dataset.cleaned_file_path),
        "metadata": dataset.get_metadata(),
        "quality_analysis": dataset.get_quality_analysis(),
        "cleaning_summary": dataset.get_cleaning_summary(),
        "audit_logs": [
            {
                "id": a.id,
                "step_number": a.step_number,
                "action_type": a.action_type,
                "target_column": a.target_column,
                "decision_details": a.decision_details,
                "rationale": a.rationale,
                "before_metrics": a.get_before_metrics(),
                "after_metrics": a.get_after_metrics(),
                "created_at": a.created_at.isoformat()
            }
            for a in dataset.audit_logs
        ]
    }

@router.get("/datasets/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, rows: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = get_current_df(dataset)
    preview_df = df.head(rows).replace({np.nan: None})
    return {
        "columns": list(df.columns),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "rows": preview_df.to_dict(orient="records")
    }

@router.post("/datasets/{dataset_id}/clean", response_model=CleaningResultResponse)
def apply_cleaning(dataset_id: str, plan: CleaningPlanRequest, db: Session = Depends(get_db)):
    """
    Executes strictly user-specified cleaning decisions.
    Never automatically mutates data without explicit user decision.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Read the base raw dataset
    raw_df, _ = parse_uploaded_file(dataset.file_path, os.path.basename(dataset.file_path))

    # Execute plan
    cleaned_df, audit_items, diff_metrics = execute_cleaning_plan(raw_df, plan)

    # Save cleaned file
    cleaned_filename = f"{dataset.id}_cleaned.csv"
    cleaned_path = os.path.join(CLEANED_DIR, cleaned_filename)
    cleaned_df.to_csv(cleaned_path, index=False)

    # Recalculate quality analysis on cleaned data
    new_quality = analyze_dataset(cleaned_df)

    # Clear previous audit logs for this dataset
    db.query(AuditLog).filter(AuditLog.dataset_id == dataset.id).delete()

    created_logs = []
    for item in audit_items:
        log = AuditLog(
            dataset_id=dataset.id,
            step_number=item["step_number"],
            action_type=item["action_type"],
            target_column=item["target_column"],
            decision_details=item["decision_details"],
            rationale=item["rationale"],
            before_metrics=json.dumps(item["before_metrics"]),
            after_metrics=json.dumps(item["after_metrics"])
        )
        db.add(log)
        created_logs.append(log)

    old_quality = dataset.get_quality_analysis()
    quality_before = old_quality.get("overall_quality_score", 0.0)
    quality_after = new_quality.get("overall_quality_score", 0.0)

    # Update dataset
    dataset.cleaned_file_path = cleaned_path
    dataset.cleaned_at = datetime.datetime.utcnow()
    dataset.status = "cleaned"
    dataset.quality_analysis_json = json.dumps(new_quality)
    dataset.cleaning_summary_json = json.dumps({
        **diff_metrics,
        "quality_score_before": quality_before,
        "quality_score_after": quality_after
    })
    db.commit()
    db.refresh(dataset)

    sample_preview = cleaned_df.head(15).replace({np.nan: None}).to_dict(orient="records")

    return CleaningResultResponse(
        dataset_id=dataset.id,
        rows_before=diff_metrics["rows_before"],
        rows_after=diff_metrics["rows_after"],
        columns_before=diff_metrics["columns_before"],
        columns_after=diff_metrics["columns_after"],
        missing_cells_before=diff_metrics["missing_cells_before"],
        missing_cells_after=diff_metrics["missing_cells_after"],
        duplicates_before=diff_metrics["duplicates_before"],
        duplicates_after=diff_metrics["duplicates_after"],
        quality_score_before=quality_before,
        quality_score_after=quality_after,
        audit_logs=[
            {
                "id": a.id or idx,
                "step_number": a.step_number,
                "action_type": a.action_type,
                "target_column": a.target_column,
                "decision_details": a.decision_details,
                "rationale": a.rationale,
                "before_metrics": a.get_before_metrics(),
                "after_metrics": a.get_after_metrics(),
                "created_at": a.created_at.isoformat() if a.created_at else datetime.datetime.utcnow().isoformat()
            }
            for idx, a in enumerate(created_logs, 1)
        ],
        sample_preview=sample_preview
    )

@router.get("/datasets/{dataset_id}/download-cleaned")
def download_cleaned_dataset(dataset_id: str, format: str = "csv", db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset or not dataset.cleaned_file_path or not os.path.exists(dataset.cleaned_file_path):
        raise HTTPException(status_code=404, detail="Cleaned dataset not available. Please apply cleaning first.")

    if format == "xlsx":
        df = pd.read_csv(dataset.cleaned_file_path)
        excel_path = os.path.join(CLEANED_DIR, f"{dataset.id}_cleaned.xlsx")
        df.to_excel(excel_path, index=False)
        return FileResponse(excel_path, filename=f"cleaned_{dataset.name}.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    return FileResponse(
        dataset.cleaned_file_path,
        filename=f"cleaned_{dataset.name}.csv" if not dataset.name.endswith(".csv") else f"cleaned_{dataset.name}",
        media_type="text/csv"
    )

@router.get("/datasets/{dataset_id}/visualizations")
def get_visualizations(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = get_current_df(dataset)
    quality = dataset.get_quality_analysis()

    recommended_charts = generate_recommended_visualizations(df, quality)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    return {
        "dataset_name": dataset.name,
        "is_cleaned": bool(dataset.cleaned_file_path),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "recommended_charts": recommended_charts
    }

@router.post("/datasets/{dataset_id}/custom-chart")
def create_custom_chart(dataset_id: str, req: ChartDataRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = get_current_df(dataset)
    spec = generate_custom_chart(
        df=df,
        chart_type=req.chart_type,
        x_col=req.x_column,
        y_col=req.y_column,
        color_col=req.color_column,
        aggregation=req.aggregation or "none",
        title=req.title
    )
    return spec

@router.post("/datasets/{dataset_id}/ask", response_model=UserQuestionResponse)
def ask_question_about_dataset(dataset_id: str, req: UserQuestionRequest, db: Session = Depends(get_db)):
    """Provides empirical, evidence-grounded answers to natural questions about the dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = get_current_df(dataset)
    quality = dataset.get_quality_analysis()
    res = answer_dataset_question(df, req.question, quality)
    return UserQuestionResponse(**res)

@router.get("/datasets/{dataset_id}/report-pdf")
def download_pdf_report(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    pdf_filename = f"report_{dataset.id}.pdf"
    output_pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    audit_logs_dicts = [
        {
            "step_number": a.step_number,
            "target_column": a.target_column,
            "action_type": a.action_type,
            "decision_details": a.decision_details,
            "rationale": a.rationale
        }
        for a in dataset.audit_logs
    ]

    generate_pdf_report(
        output_pdf_path=output_pdf_path,
        dataset_name=dataset.name,
        quality_analysis=dataset.get_quality_analysis(),
        cleaning_summary=dataset.get_cleaning_summary(),
        audit_logs=audit_logs_dicts
    )

    return FileResponse(
        output_pdf_path,
        filename=f"Data_Intelligence_Report_{dataset.name.replace('.csv','')}.pdf",
        media_type="application/pdf"
    )
