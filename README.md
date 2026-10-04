# AnalytiX: AI-Powered Data Quality & Analytics Platform

**AnalytiX** is a comprehensive, interactive data quality, cleaning, exploratory analytics, visualization, and executive reporting platform built with **React, TypeScript, Tailwind CSS, FastAPI, Pandas, Plotly.js, and ReportLab**.

The platform provides a guided 6-stage workflow that takes raw tabular data and transforms it into verified, clean datasets accompanied by empirical insights, custom interactive visualizations, and formal downloadable PDF reports.

---

## 🏛️ Platform Architecture & 6-Stage Workflow

```
[1. Upload] ➔ [2. Data Inspection] ➔ [3. Data Cleaning] ➔ [4. AI Intelligence] ➔ [5. Visualization and Insights] ➔ [6. Data Intelligence and Report]
```

### 1. Upload
* **Multi-Format Ingestion**: Supports drag-and-drop and file selection for **CSV, Excel (`.xlsx`, `.xls`), HTML tables, PDF tables, and Graphviz DOT (`.dot`, `.gv`)** files.
* **Instant Benchmark Datasets**: Includes pre-configured enterprise datasets for immediate evaluation:
  * **Retail & E-commerce Sales** (600 rows — multi-channel revenue, regional discounts, outliers)
  * **B2B SaaS Revenue & Churn** (500 rows — ARR cohorts, feature adoption, churn risk)
  * **Commercial Credit & Loan Risk** (550 rows — credit scores, DTI ratios, delinquency)
* **Session Management**: Real-time format detection, row and column parsing, and active dataset switching.

### 2. Data Inspection
* **Structural Dimension Metrics**: Immediate visibility into total rows, columns (numeric vs. categorical breakdown), total cells, and duplicate row counts with percentages.
* **Missingness Density**: Quick identification of total missing cells and overall dataset missingness percentage.
* **Column-by-Column Profiling**: Concise profiling table presenting each column's name, inferred data type, missing cell count, visual missing percentage progress indicator, and unique value cardinality.

### 3. Data Cleaning
* **Strict User-Directed Control**: Changes are never applied automatically or silently. Every transformation requires explicit user decision and applies only when confirmed.
* **Whitespace Trimming**: Identifies and removes irregular leading, trailing, and redundant whitespace across column headers and text values.
* **Data Type Inconsistency Correction**: Automatically flags mismatched data types (integers, floats, dates, or booleans stored as text), suggests appropriate target types, and applies conversion.
* **Missing Value Imputation**: Flexible null handling strategies (Median, Mean, Mode, Constant, Forward Fill, Backward Fill) or row/column removal.
* **Outlier Handling**: Configurable 1.5× IQR fence clipping (Winsorization) or record exclusion.
* **Deduplication**: One-click detection and removal of exact duplicate rows.
* **Immutable Audit Trail**: Automatically logs every executed action with step number, target column, action type, rationale, and before/after metric deltas.
* **Cleaned Data Download**: One-click export of the cleaned dataset in CSV format.

### 4. AI Intelligence
* **Multi-Dimensional Quality Scoring**: Comprehensive health assessment (0–100 overall score) evaluating Completeness, Uniqueness, Consistency, and Validity.
* **Statistical Anomaly Detection**: Highlights extreme values via the 1.5× Interquartile Range (IQR) rule and flags inconsistent data types.
* **Evidence-Grounded Data Q&A**: Interactive query engine answering natural language questions with real calculations, aggregations, groupings, comparisons, rankings, and filtering performed directly against dataset records.
* **Sequential Procedural Steps**: When requested for operational procedures or how-to steps, produces strictly sequential, numbered instructions without conversational filler.
* **Hallucination Prevention**: Identifies unanswerable questions and explicitly indicates missing information rather than hallucinating answers.
* **Analytical Discoveries**: Automatically surfaces key variable relationships and recommended next investigative avenues.

### 5. Visualization and Insights
* **Automated Recommendations**: Generates up to 6 insight-driven charts chosen based on column distributions, variance, and analytical utility.
* **Supported Chart Types**: Categorical Bar charts, Distribution Histograms, Bivariate Scatter plots, Quartile Box plots, Line trends, Donut/Pie compositions, and Correlation Heatmaps.
* **Interactive Custom Chart Builder**: Allows users to configure chart types, X/Y axes, color dimensions, and aggregation methods (Sum, Mean, Count, or None).
* **Plotly.js Interactivity**: High-performance rendering with interactive pan, zoom, hover tooltips, customizable corporate color palettes, and PNG export.

### 6. Data Intelligence and Report
* **Executive Summary**: Synthesizes verified record counts, feature totals, overall quality scores, and cleaning deltas into an executive overview.
* **Full Transformation Audit Log**: Complete tabular record documenting every cleaning operation, affected column, and business rationale.
* **Publication-Ready PDF Export**: Generates and downloads a formal, styled PDF report via ReportLab containing executive summaries, quality indices, transformation diffs, and audit trails.

---

## 🛠️ Technologies Used

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Lucide React, Plotly.js (`react-plotly.js`), Axios |
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pandas, NumPy, SciPy, ReportLab, SQLAlchemy, Pydantic |
| **Database & Storage** | SQLite (default `dataintel.db`), Local file storage (`storage/uploads`, `storage/cleaned`, `storage/reports`) |

---

## 📁 Project Structure

```
AnalytiX/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI route handlers
│   │   ├── core/           # Configuration & database engine
│   │   ├── models/         # SQLAlchemy database models
│   │   ├── schemas/        # Pydantic validation schemas
│   │   ├── services/       # Analysis, cleaning, QA, viz, and PDF services
│   │   └── main.py         # Application entrypoint & lifespan
│   ├── storage/            # Uploads, cleaned outputs, and generated reports
│   ├── requirements.txt    # Python dependencies
│   ├── test_backend.py     # Backend service unit tests
│   └── test_api_integration.py # API integration tests
├── frontend/
│   ├── src/
│   │   ├── components/     # UI components (Navbar, Header, Charts, Audit table)
│   │   ├── pages/          # 6 workflow stage pages
│   │   ├── services/       # Axios API client
│   │   ├── types/          # TypeScript interfaces
│   │   ├── App.tsx         # Main application shell & workflow state
│   │   └── main.tsx        # React entrypoint
│   ├── package.json        # Frontend dependencies & scripts
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   └── vite.config.ts      # Vite configuration
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
* **Python**: 3.10 or higher
* **Node.js**: 18.x or higher and **npm**

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
* **Interactive API Docs (Swagger)**: `http://localhost:8000/docs` (or `http://localhost:8000/api/docs`)

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
# From the backend directory
cd backend
python -m unittest test_backend.py test_api_integration.py
```
