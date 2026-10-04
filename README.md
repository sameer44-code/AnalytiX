# AnalytiX: AI Data Intelligence & Governance Platform

An intelligent, autonomous data quality, cleaning, and analytics platform built with **React, TypeScript, Tailwind CSS, FastAPI, Pandas, Plotly.js, and ReportLab**.

---

## 🏛️ Platform Architecture & 6-Stage Workflow

```
[1. Upload] ➔ [2. Data Inspection] ➔ [3. Data Cleaning] ➔ [4. AI Intelligence] ➔ [5. Visualization & Insights] ➔ [6. Data Intelligence & Report]
```

### 1. Upload
* **Universal Ingestion**: Supports **CSV, Excel (`.xlsx`, `.xls`), and TSV** files.
* **Instant Demo Datasets**: Built-in sample datasets (Retail Sales, SaaS Churn, Financial Risk) for immediate exploration.
* **Streamlined UI**: Drag-and-drop file upload with real-time format detection and validation.

### 2. Data Inspection
* **Quality Scoring**: Multi-dimensional quality evaluation (0–100) combining Completeness, Uniqueness, Consistency, and Validity.
* **Column Profiling**: Detailed per-column statistics including non-null counts, null percentages, distinct cardinalities, data types, min/max/mean/std, and preview samples.
* **Outlier Detection**: Automated statistical outlier detection using the Interquartile Range (1.5× IQR) rule.
* **Correlation Matrix**: Dynamic cross-feature correlation calculation for numerical variables.

### 3. Data Cleaning
* **Trim Spaces**: Automated detection of irregular leading, trailing, and unnecessary whitespace across column names and text values with one-click cleanup.
* **Data Type Inconsistency Detection**: Automatically inspects column values to identify mismatched data types (integers, decimals, dates, or booleans stored as text) with suggested target types and one-click conversion.
* **Missing Value Imputation**: Flexible null handling (Mean, Median, Mode, Constant, Forward Fill, Backward Fill) or row/column removal.
* **Outlier Treatment**: Winsorization/clipping to IQR fences or record elimination.
* **Deduplication**: Identification and one-click removal of exact duplicate rows.
* **Audit Trail**: Every applied cleaning action generates an immutable audit record with step number, target column, action type, rationale, and before/after metrics.
* **Cleaned Data Download**: One-click export of the cleaned dataset in CSV format.

### 4. AI Intelligence & Data Q&A
* **Data-Backed Answers**: Analyzes actual columns and records to execute dynamic calculations, aggregations, groupings, comparisons, rankings, and filtering.
* **Steps-Only Procedures**: When asked for procedural steps or instructions, produces strictly sequential, numbered steps without conversational filler or unrelated analysis.
* **Unanswerable Query Handling**: If a question cannot be answered from the available dataset, clearly states that the required information is not available rather than hallucinating.
* **Dynamic & Agnostic**: Operates on any uploaded dataset without hardcoded domain or industry assumptions.

### 5. Visualization and Insights
* **Insight-Driven Chart Recommendations**: Visualizations are automatically selected to communicate meaningful analytical insights from dataset columns.
* **Supported Visual Types**: Categorical aggregations (Bar), bivariate relationships (Scatter), distribution histograms, quartile Box plots, trends (Line), compositions (Pie), and correlation Heatmaps.
* **Intelligent Axis & Aggregation**: Meaningful X/Y axis assignment with appropriate aggregations (Sum, Mean, Count).
* **Interactive Exploration**: Powered by Plotly.js with interactive zoom, pan, hover tooltips, customizable color palettes, and PNG export.

### 6. Data Intelligence and Report
* **Executive Summary**: Synthesizes dataset dimensions, overall quality index, anomaly logs, and cleaning history.
* **Compliance Audit Log**: Complete step-by-step audit table documenting every data transformation.
* **Executive PDF Export**: Server-side ReportLab PDF generator creating publication-ready downloadable compliance reports.

---

## 🛠️ Technologies Used

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Lucide React, Plotly.js (`react-plotly.js`), Axios |
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pandas, NumPy, SciPy, ReportLab, SQLAlchemy |
| **Database & Storage** | SQLite (default `dataintel.db` with zero configuration required), Local file storage |

---

## 🚀 How to Run the Project

### 1. Prerequisites
* Python 3.10+
* Node.js 18+ and npm

### 2. Backend Setup & Launch
```bash
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* **API Base URL**: `http://localhost:8000`
* **Health Check**: `http://localhost:8000/api/health`
* **Interactive API Docs (Swagger)**: `http://localhost:8000/api/docs`

### 3. Frontend Setup & Launch
```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
* **Application Web UI**: `http://localhost:5173`

### 4. Running Backend Tests
```bash
cd backend
python -m unittest test_backend.py
python -m unittest test_api_integration.py
```
