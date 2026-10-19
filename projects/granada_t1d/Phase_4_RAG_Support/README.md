# Phase 4: Grounded RAG Clinical Decision Support Engine

> Current location: `Phase_4_RAG_Support`. From the BTP root, run `python scripts/run_phase_4.py`. These commands regenerate original research outputs. See [project structure](../docs/PROJECT_STRUCTURE.md) and [the audit](../docs/reports/PROJECT_AUDIT_REPORT.md).

## Overview
Phase 4 builds a Retrieval-Augmented Generation (RAG) system with a **FAISS vector index** of the official **American Diabetes Association (ADA) Standards of Medical Care in Diabetes**. It ingests the simulated 60-day trajectories from the generative models, retrieves relevant clinical protocols, and synthesizes an explainable, physician-grade quarterly care consultation report.

---

## Scripts & Usage

### 1. `ada_diabetes_clinical_guidelines.json`
* **Content**: Official clinical protocols covering Glycemic targets (AGP), Nocturnal hypoglycemia mitigation, Nephropathy (kidney) monitoring, Retinopathy (eye) care, Cardiovascular/lipid therapy, and Poly-autoimmune thyroid screening.

### 2. `rag_clinical_engine.py`
* **Purpose**:
  - Builds and saves `ada_guidelines_faiss.index`
  - Ingests a patient's 60-day simulated trajectory from the trained VAE/cGAN
  - Queries FAISS for matching ADA clinical rules
  - Generates a structured Chain-of-Thought (CoT) doctor consultation report with zero hallucination
* **Execution**: `python3 rag_clinical_engine.py`
* **Outputs**:
  - `ada_guidelines_faiss.index` (FAISS vector database)
  - `doctor_clinical_decision_report_sample.txt` (Physician consultation report)

---

## Report Structure
1. **Section 1**: 60-Day Generative Trajectory Prognosis (Projected Mean Glucose, TIR, TBR, CV, and HbA1c).
2. **Section 2**: RAG Evidence Retrieval with similarity relevance scores and ADA chapter citations.
3. **Section 3**: Evidence-Grounded Physician Action Plan (e.g., basal insulin titration, microalbuminuria lab orders, retinopathy screening).
