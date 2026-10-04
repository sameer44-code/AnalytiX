import datetime
import json
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, default=0)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    status = Column(String, default="uploaded")  # uploaded, analyzed, cleaned
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Serialized JSON fields (Text for universal DB compatibility)
    metadata_json = Column(Text, default="{}")
    quality_analysis_json = Column(Text, default="{}")
    
    # Cleaning state
    cleaned_file_path = Column(String, nullable=True)
    cleaned_at = Column(DateTime, nullable=True)
    cleaning_summary_json = Column(Text, default="{}")
    
    # Audit log relationship
    audit_logs = relationship("AuditLog", back_populates="dataset", cascade="all, delete-orphan", order_by="AuditLog.step_number")

    def get_metadata(self):
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except Exception:
            return {}

    def get_quality_analysis(self):
        try:
            return json.loads(self.quality_analysis_json) if self.quality_analysis_json else {}
        except Exception:
            return {}

    def get_cleaning_summary(self):
        try:
            return json.loads(self.cleaning_summary_json) if self.cleaning_summary_json else {}
        except Exception:
            return {}


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_id = Column(String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    step_number = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)  # fill, remove_rows, remove_column, keep, handle_outliers, remove_duplicates
    target_column = Column(String, nullable=True)
    decision_details = Column(String, nullable=False)
    rationale = Column(String, nullable=False)
    before_metrics = Column(Text, default="{}")
    after_metrics = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="audit_logs")

    def get_before_metrics(self):
        try:
            return json.loads(self.before_metrics) if self.before_metrics else {}
        except Exception:
            return {}

    def get_after_metrics(self):
        try:
            return json.loads(self.after_metrics) if self.after_metrics else {}
        except Exception:
            return {}
