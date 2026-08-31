# Pima Indians Diabetes ML + RAG + LLM Pipeline

> **Educational / Research System — Not a Medical Diagnostic Tool**

This project demonstrates a complete research pipeline combining:
- **SVM (RBF) machine-learning prediction** on the Pima Indians Diabetes dataset
- **Retrieval-Augmented Generation (RAG)** using real medical documents from NIDDK and CDC
- **OpenAI LLM explanation** grounded in retrieved evidence with citation validation

---

## Architecture

```
PIMA PATIENT INPUT (8 features)
         │
         ▼
  INPUT VALIDATION
  (type, range, zero-as-missing)
         │
         ▼
 EXISTING PREPROCESSING
  KNNImputer → StandardScaler
         │
         ▼
    SVM (RBF) MODEL
    (diabetes_best_model.joblib)
         │
    ┌────┴────┐
    ▼         ▼
 CLASS      MODEL
 LABEL    PROBABILITY
    └────┬────┘
         │
         ▼
 LOCAL SENSITIVITY ANALYSIS
 (deterministic ±1σ perturbation)
         │
         ▼
    RAG QUERY
 (context-aware query from features)
         │
         ▼
   FAISS SEARCH
   (BAAI/bge-base-en-v1.5)
         │
         ▼
 RETRIEVED MEDICAL CHUNKS
 (NIDDK + CDC documents)
         │
         ▼
   OPENAI LLM (Responses API)
 (chunk_id citations only)
         │
         ▼
  CITATION VALIDATION
  (server-side: reject non-retrieved IDs)
         │
         ▼
 STRUCTURED EXPLANATION
 (summary / factors / sources / disclaimer)
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| **SVM (RBF)** | Dataset-class prediction and model probability |
| **KNNImputer + StandardScaler** | Preprocessing (fit on training data) |
| **FAISS RAG** | Retrieval of real medical reference material |
| **LLM** | Grounded natural-language explanation only |

The LLM **cannot** diagnose, prescribe, or fabricate citations.

---

## Important Constraints

- This system is **NOT** a medical diagnostic tool and has **not been clinically validated**.
- The ML output is solely a **research-dataset label**, not a medical probability or diagnosis.
- Feature input ranges are restricted to the training dataset domain to prevent out-of-distribution machine-learning errors. They are application safety limits.
- The LLM explanation is grounded in **retrieved documents only** — fabricated citations are blocked server-side.
- All medical sources have full provenance (publisher, URL, checksum, retrieval date).

---

## Installation

```bash
# Clone and enter the project directory
cd "PIma - BTP"

# Install from the pinned lock file (reproducible, exact versions)
pip install -r requirements.lock
```

> **Note**: Do NOT use `requirements.txt` for setup — it uses loose `>=` ranges and
> produces non-reproducible environments. `requirements.lock` pins exact versions.

---

## Fresh Clone Setup

After cloning, several files must be generated before the API will start.
Run these steps **in order**:

```bash
# 1. Generate ML artifacts (requires diabetes.csv to be present)
python scripts/rebuild_legacy_artifacts.py --force

# 2. Verify artifacts (writes artifacts_manifest.json)
python scripts/artifact_gate.py

# 3. Download medical sources from NIDDK and CDC
python scripts/download_medical_sources.py

# 4. Build FAISS vector index
python scripts/build_index.py

# 5. Start the backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

> **Important**: Steps 1–4 must complete successfully before step 5.
> `.joblib` files, downloaded `.txt` documents, `source_manifest.json`,
> and `vector_store/` are all excluded from version control (see `.gitignore`).
> They must be regenerated on every fresh clone.

---

## Environment Setup

```bash
# Copy the example file
copy .env.example .env
```

Edit `.env` and fill in:
```
OPENAI_API_KEY=sk-...       # Your rotated OpenAI API key
OPENAI_MODEL=gpt-4o-mini    # Or gpt-4o
```

**Never commit `.env` to version control.**

This re-runs the exact Notebook Cell 1 logic with `RANDOM_STATE=42` and verifies:
- Selected model is `SVM (RBF)`
- Recall ≥ 0.88, F1 ≥ 0.79, AUC ≥ 0.89

### 2. Verify Artifacts (Phase 0 gate)

```bash
python scripts/artifact_gate.py
```

Output: `[GATE PASSED] All artifacts verified.`

### 3. Download Medical Sources

```bash
python scripts/download_medical_sources.py
```

Downloads 7–8 pages from NIDDK and CDC into `data/medical_docs/` with SHA-256 checksums.
All sources are U.S. Government public domain educational content.

### 4. Build FAISS Index

```bash
python scripts/build_index.py
```

Creates `vector_store/faiss.index` and `vector_store/index_manifest.json`.
The index records a hash of the source manifest — if sources change, the index is automatically invalidated.

### 5. Start the Backend

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 6. Open the Frontend

Open `frontend/index.html` in your browser (via a local file server or VS Code Live Server).

---

## API Reference

### `GET /api/health`
Returns service status.

```json
{ "status": "ok", "ml_ready": true, "rag_ready": true }
```

### `POST /api/diabetes/analyze`

**Request body:**
```json
{
  "Pregnancies": 2,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 120,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Success response:**
```json
{
  "prediction": 1,
  "model_probability": 0.78,
  "risk_label": "Model predicts positive diabetes class",
  "explanation_status": "success",
  "explanation": {
    "summary": "...",
    "prediction_explanation": "...",
    "important_factors": [ { "factor": "Glucose", "explanation": "...", "citation_chunk_ids": ["..."] } ],
    "medical_context": "...",
    "recommendation": "...",
    "sources": [ { "chunk_id": "...", "source": "...", "publisher": "NIDDK", "page": 1, "url": "..." } ],
    "disclaimer": "..."
  },
  "feature_sensitivity": {
    "values": { "Glucose": 0.12, "BMI": 0.08, ... },
    "label": "Local model sensitivity estimate; not proof of medical causation."
  }
}
```

**Partial failure (LLM unavailable — ML still returned):**
```json
{
  "prediction": 1,
  "model_probability": 0.78,
  "risk_label": "Model predicts positive diabetes class",
  "explanation_status": "unavailable",
  "explanation_message": "ML prediction successful; AI explanation service temporarily unavailable.",
  "sources": []
}
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Artifact gate verification
- Prediction service (valid inputs, all invalid types, zero-as-missing, upper bounds)
- RAG (chunking, loading, embeddings, vector store, stale index detection)
- LLM (citation validation, fabricated citation blocking, API failure fallback)
- End-to-end API flow (mocked LLM — no real API calls)

---

## RAG Evaluation

After building the index:
```bash
python scripts/evaluate_rag.py
```

Reports:
- Recall@K for expected document IDs
- Chunk ID validity rate
- Empty-context trigger rate

---

## Medical Sources

All sources are downloaded from U.S. Government public domain sources:

| ID | Publisher | Title |
|---|---|---|
| `niddk-diabetes-overview` | NIDDK | What Is Diabetes? |
| `niddk-diabetes-risk` | NIDDK | Risk Factors for Type 2 Diabetes |
| `niddk-insulin-resistance` | NIDDK | Insulin Resistance and Prediabetes |
| `niddk-diabetes-prevention` | NIDDK | Preventing Type 2 Diabetes |
| `cdc-diabetes-risk-factors` | CDC | Diabetes Risk Factors |
| `cdc-diabetes-symptoms` | CDC | Diabetes Symptoms |
| `cdc-gestational-diabetes` | CDC | Gestational Diabetes |

To add more sources, add entries to `data/medical_docs/source_registry.json` and re-run the downloader and index builder.

---

## Security Notes

- `OPENAI_API_KEY` is loaded only from `.env` — never from source code, logs, or frontend
- `.env` is gitignored
- API responses never contain stack traces, file paths, or configuration details
- CORS is restricted to configured frontend origins (`null` origin is explicitly excluded)
- Rate limited to 10 requests/minute per IP
- LLM citation IDs are validated server-side — fabricated references are rejected
- Retrieved document chunks are delimited as untrusted data in the LLM prompt

> **Rate limiting behind a reverse proxy**: The default `slowapi` configuration uses
> the request's remote address. Behind a proxy (nginx, Caddy, etc.), all requests may
> appear to come from the proxy IP. Configure `FORWARDED_ALLOW_IPS` or use
> `get_remote_address` with proper trusted-proxy settings before production deployment.

---

## Disclaimer

This system provides an AI-assisted explanation of a machine-learning prediction for **educational and research purposes only**. It is **not a medical diagnosis** and does not replace advice from a qualified healthcare professional. Always consult a licensed medical provider for health decisions.
