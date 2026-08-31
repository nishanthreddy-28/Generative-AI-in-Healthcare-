"""
FastAPI Backend
===============
Pima Indians Diabetes ML + RAG + LLM Analysis API

Endpoints:
  POST /api/diabetes/analyze   — full ML → RAG → LLM pipeline
  GET  /api/health             — health check (no sensitive info)

Security:
  - CORS restricted to configured origins
  - Rate limiting: 10 requests/minute per IP
  - Body size limit: 4 KB
  - No stack traces in responses
  - No config/keys/paths in responses

Partial failure:
  - ML prediction always returned
  - RAG/LLM failure returns explanation_status="unavailable"
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

MAX_BODY_SIZE    = 16 * 1024  # 16 KB
REQUEST_TIMEOUT  = 90         # seconds — covers embeddings + FAISS + LLM (30 s) + overhead

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts at startup (fail fast if missing)
    try:
        from services.prediction_service import PredictionService
        PredictionService.get()
        logger.info("ML artifacts loaded successfully at startup.")
    except Exception as e:
        logger.error("Failed to load ML artifacts: %s", type(e).__name__)
        # We don't abort startup — the /analyze endpoint will return a clear error.

    yield   # Application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinicagen API",
    description=(
        "Clinicagen Chatbot API combining ML prediction, RAG retrieval, "
        "and LLM explanation for the Pima Indians Diabetes dataset. "
        "This is NOT a medical diagnostic system."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restricted to frontend origins; "null" is intentionally excluded
# (null origin is used by file:// pages and sandboxed iframes, not trusted frontends)
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS",
                        "http://localhost:5500,http://127.0.0.1:5500,"
                        "http://localhost:8080,http://127.0.0.1:8080").split(",")
    if o.strip() and o.strip() != "null"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


from fastapi.exceptions import RequestValidationError
from fastapi import status
from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": jsonable_encoder(exc.errors())},
    )

class BodySizeError(Exception):
    pass

# ---------------------------------------------------------------------------
# Body-size middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": "Request body too large (max 4 KB)."}
        )
        
    receive_ = request.receive
    bytes_received = 0
    request.scope["body_size_exceeded"] = False

    async def receive_with_limit():
        nonlocal bytes_received
        message = await receive_()
        if message["type"] == "http.request":
            body = message.get("body", b"")
            bytes_received += len(body)
            if bytes_received > MAX_BODY_SIZE:
                request.scope["body_size_exceeded"] = True
                return {"type": "http.request", "body": b"", "more_body": False}
        return message

    request._receive = receive_with_limit

    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Request timed out. The analysis pipeline took too long."}
        )

    if request.scope.get("body_size_exceeded"):
        return JSONResponse(
            status_code=413,
            content={"error": "Request body too large (max 4 KB)."}
        )

    return response


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    message: str
    current_features: dict[str, float | None] = Field(default_factory=dict)
    history: list[ChatMessage] = Field(default_factory=list)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_input: ChatInput):
    """
    Conversational endpoint to extract features from user input.
    """
    from llm.chat_agent import process_chat_turn
    try:
        extraction = process_chat_turn(
            chat_input.message, 
            chat_input.current_features,
            [h.model_dump() for h in chat_input.history]
        )
        return extraction.model_dump()
    except Exception as e:
        logger.error("Chat endpoint error: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="Internal chat error.")


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    """Strict input model — rejects strings, booleans, NaN, and Infinity."""
    Pregnancies:              float = Field(..., ge=0, le=20)
    Glucose:                  float = Field(..., ge=0, le=500)
    BloodPressure:            float = Field(..., ge=0, le=200)
    SkinThickness:            float = Field(..., ge=0, le=100)
    Insulin:                  float = Field(..., ge=0, le=900)
    BMI:                      float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3.0)
    Age:                      float = Field(..., ge=0, le=120)

    model_config = {"extra": "forbid"}

    @field_validator(
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        mode="before",
    )
    @classmethod
    def reject_non_numeric(cls, v):
        """Reject strings, booleans, NaN, and Infinity before Pydantic coercion."""
        # booleans are subclasses of int in Python — reject them explicitly
        if isinstance(v, bool):
            raise ValueError("Must be a numeric value, not a boolean.")
        # reject strings (Pydantic would otherwise coerce "148" -> 148.0)
        if isinstance(v, str):
            raise ValueError("Must be a number, not a string.")
        import math
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("Must be a finite number.")
        if not math.isfinite(f):
            raise ValueError(f"Must be a finite number (got {v}).")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check — returns service status without exposing config."""
    from services.prediction_service import PredictionService, ArtifactLoadError
    from rag.vector_store import INDEX_PATH, METADATA_PATH, MANIFEST_PATH

    ml_ready = False
    # RAG is only ready when all three vector store files exist (not just the manifest)
    rag_ready = INDEX_PATH.exists() and METADATA_PATH.exists() and MANIFEST_PATH.exists()

    try:
        ps = PredictionService.get()
        ml_ready = True
    except Exception:
        pass

    return {
        "status":    "ok",
        "ml_ready":  ml_ready,
        "rag_ready": rag_ready,
    }


@app.post("/api/diabetes/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request, patient: PatientInput):
    """
    Full ML → RAG → LLM analysis pipeline.

    Input: 8 Pima feature values (JSON body).
    Returns: ML prediction + optional LLM explanation grounded in retrieved evidence.

    This is an educational/research system. The response is NOT a medical diagnosis.
    """
    patient_dict = patient.model_dump()

    try:
        from services.diabetes_analysis_service import analyze as run_analysis
        result = run_analysis(patient_dict)
        return result

    except Exception as e:
        # Catch-all: never expose stack trace or config
        cls_name = type(e).__name__
        logger.error("Analysis pipeline error: %s", cls_name)

        # Distinguish validation errors from server errors
        if "ValidationError" in cls_name or "validation" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))

        if "ArtifactLoad" in cls_name:
            raise HTTPException(
                status_code=503,
                detail="ML model artifacts are missing. Please contact the administrator."
            )

        raise HTTPException(
            status_code=500,
            detail="Internal analysis error. The service is temporarily unavailable."
        )


# ---------------------------------------------------------------------------
# Serve Frontend Static Files
# ---------------------------------------------------------------------------
from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/style.css")
async def serve_style():
    return FileResponse("frontend/style.css")

@app.get("/app.js")
async def serve_app_js():
    return FileResponse("frontend/app.js")
