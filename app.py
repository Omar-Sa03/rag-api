from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import structlog
import time

from logging_utils import configure_logging
from exceptions import RAGException
from api_v1 import router as api_v1_router
from limiter import limiter

# Configure logging
configure_logging()
logger = structlog.get_logger()

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with hybrid search and re-ranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Middleware ---

# CORS
origins = ["*"] # Configure appropriately for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Generate request ID
    request_id = request.headers.get("X-Request-ID", str(time.time()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response

# --- Exception Handlers ---

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": exc.__class__.__name__}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

# --- Routers ---

app.include_router(api_v1_router, prefix="/v1", tags=["v1"])
# For now, V2 can point to V1 or be a separate router
app.include_router(api_v1_router, prefix="/v2", tags=["v2"])

# --- Health & Metrics ---

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": time.time()
    }

# Initialize Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
async def startup_event():
    logger.info("application_startup")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("application_shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
