from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Data Analysis API")
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Add log_level="info" to see your logger messages
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir="src",
        log_level="info"
    )