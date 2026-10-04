import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import engine, Base, SessionLocal
from app.api.routes import router as api_router
from app.models.dataset import Dataset
from app.services.sample_data import SAMPLE_CATALOG
from app.services.data_analyzer import analyze_dataset
from app.core.config import UPLOAD_DIR
import uuid
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    
    # Automatically pre-seed sample datasets if database is empty
    db = SessionLocal()
    try:
        if db.query(Dataset).count() == 0:
            for key, meta in SAMPLE_CATALOG.items():
                df = meta["generator"]()
                dataset_id = str(uuid.uuid4())
                save_filename = f"{dataset_id}_{key}.csv"
                save_path = os.path.join(UPLOAD_DIR, save_filename)
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
                        "sample_key": key
                    }),
                    quality_analysis_json=json.dumps(quality)
                )
                db.add(dataset)
            db.commit()
    except Exception as e:
        print(f"[Warning] Sample pre-seeding notice: {e}")
    finally:
        db.close()

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "frontend": "http://localhost:5173",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/api/health"
    }

@app.get("/api/docs", include_in_schema=False)
def api_docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
