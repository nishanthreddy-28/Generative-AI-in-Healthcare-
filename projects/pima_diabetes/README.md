# Pima diabetes · Clinicagen

An SVM diabetes classifier with a conversational interface and medical-source explanations.

Clinicagen collects the eight Pima input features, runs a saved scikit-learn model, and shows the prediction alongside a local feature-sensitivity analysis. The analysis endpoint retrieves relevant medical passages and can use Groq to compose a structured explanation with source references.

The dataset has **768 records**, eight input features, and a binary `Outcome` label. This is a cross-sectional classification experiment. Its output is a model classification, not a diagnosis or a forecast of disease progression.

## How the analysis works

```text
Eight numeric inputs
    → validation and zero-as-missing handling
    → saved KNN imputer and standard scaler
    → SVM classification and local sensitivity
    → medical-source retrieval
    → structured explanation with citations
```

The medical corpus is built from sources listed in [`source_registry.json`](data/medical_docs/source_registry.json). Retrieval uses BGE embeddings and a local FAISS index. The frontend is plain HTML, CSS, and JavaScript; the backend is FastAPI.

## Running Clinicagen

Run commands from `projects/pima_diabetes` in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` and copy `.env.example` using your shell's copy command. Set `GROQ_API_KEY` in `.env` for conversational input collection and LLM explanations. Keep the key out of Git.

Prepare local artifacts on a fresh checkout:

```bash
python scripts/rebuild_legacy_artifacts.py
python scripts/artifact_gate.py
python scripts/download_medical_sources.py
python scripts/build_index.py
```

The rebuild command fits models to reproduce the historical experiment; it is not just a file download. It checks the selected model/metrics before writing artifacts. The source and embedding steps need network access on first setup. Model binaries, downloaded source documents, and the index are excluded from Git.

Start the backend:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal, from the same project directory, serve the frontend:

```bash
python -m http.server 5500 --directory frontend
```

Open [localhost:5500](http://localhost:5500). API documentation is available at [localhost:8000/api/docs](http://localhost:8000/api/docs). If you change ports, update the frontend API address and configured CORS origins together.

## Files

- `Pima.ipynb` contains the classification experiments.
- `services/` handles prediction, sensitivity, and the analysis workflow.
- `rag/` loads, chunks, embeds, and retrieves medical passages.
- `llm/` handles conversational feature collection and structured explanations.
- `frontend/` contains the browser interface.
- `scripts/` rebuilds/verifies artifacts and evaluates retrieval.
- `tests/` covers API behavior, schemas, prediction, and retrieval components.

## Evaluation status

The saved metadata records 84.4% test accuracy, 0.80 F1, and 0.898 ROC AUC for the historical SVM experiment. The rebuild script chooses the best model using that same test set, so these are development results rather than an untouched final evaluation. Model selection needs a separate validation/CV protocol before publication.

The chat collector also has a known issue: it can substitute a positive label and 80.8% when prediction context is missing. General follow-up chat does not itself retrieve evidence. These paths need correction before describing the whole conversation as grounded in model results and sources.

Run the full suite after preparing its required assets:

```bash
python -m pytest tests -q
```

An offline subset passed 49 tests on 19 October 2026; 22 were skipped for missing assets and four real embedding tests were excluded. A full fresh-install/end-to-end pass is not claimed.

## Data and use

`diabetes.csv` is the existing dataset copy used by the notebooks. Its exact source and prior preprocessing need provenance documentation before reuse in a paper. The model's training population limits how its outputs should be interpreted. This project is for research and educational use.
