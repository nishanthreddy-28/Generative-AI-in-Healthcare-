# Generative AI in Healthcare

Diabetes research across three datasets, covering classification, medical report explanation, and glucose forecasting.

The work started with tabular diabetes prediction and developed into a longitudinal project using T1DiabetesGranada. Each dataset has a separate folder because the patient populations, inputs, and prediction tasks are different. The Granada project also includes **ClearCare**, a local application for reviewing medical reports and exploring experimental glucose simulations.

[Granada project](projects/granada_t1d) · [Pima application](projects/pima_diabetes) · [Early-stage notebook](projects/early_stage_diabetes) · [Research audit](projects/granada_t1d/docs/reports/RESEARCH_AUDIT_2026-10-19.md)

## Projects

### [T1DiabetesGranada](projects/granada_t1d)

The main longitudinal research project. It uses continuous glucose monitoring records to learn a representation of 30 observed days and forecast daily glucose metrics over the next 60 days. The project contains the original multimodal VAE/GAN experiments, a separate CGM-only VAE/GRU release, and the ClearCare report assistant.

**Built with:** PyTorch, pandas, scikit-learn, Streamlit, and local document extraction.

### [Pima diabetes](projects/pima_diabetes)

An eight-feature diabetes classifier connected to **Clinicagen**, a FastAPI application with a chat interface. The analysis workflow combines an SVM prediction, local feature sensitivity, retrieval from medical sources, and an optional LLM explanation.

**Built with:** scikit-learn, FastAPI, FAISS, BGE embeddings, and Groq.

### [Early-stage diabetes](projects/early_stage_diabetes)

A notebook comparing random forest, decision tree, and logistic regression models on a 520-record symptom dataset. This was previously stored as `pre - diabetic`; the folder name now reflects the dataset's binary classification task.

**Built with:** pandas and scikit-learn.

## Getting started

```bash
git clone https://github.com/nishanthreddy-28/Generative-AI-in-Healthcare-.git
cd Generative-AI-in-Healthcare-
```

Choose a project and follow its setup instructions. There is no shared environment: install dependencies and run commands from that project's directory.

For the Granada report assistant:

```bash
cd projects/granada_t1d
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/app.txt
python -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). On Windows, activate the environment with `.venv\Scripts\activate`.

Report explanation can run without the Granada patient dataset. Forecasting needs authorized local data/model artifacts; see [data access](projects/granada_t1d/DATA_ACCESS.md). The Pima chatbot has its own [setup and API configuration](projects/pima_diabetes#running-clinicagen).

## Repository layout

```text
projects/
  granada_t1d/            Longitudinal research and ClearCare
  pima_diabetes/         Pima classifier and Clinicagen
  early_stage_diabetes/ Symptom-classification notebook
docs/
  dataset_catalog.md    Dataset sources and task descriptions
  architecture/         Earlier project diagrams
```

## Results and research notes

The corrected Granada release was evaluated on 39 held-out patients. Its glucose MAE was **23.97 mg/dL**, compared with **23.33 mg/dL** for a 30-day mean baseline. These results, the preprocessing checks, and the limitations are documented in the [model card](projects/granada_t1d/docs/models/CGM_MODEL_CARD.md) and [research audit](projects/granada_t1d/docs/reports/RESEARCH_AUDIT_2026-10-19.md).

The older experiments are retained for reference. Some need revised preprocessing or evaluation before their results can be used in a paper. The applications are research prototypes and are not clinically validated.

## Data and license

The Pima and early-stage CSVs remain with their respective projects. Dataset provenance and attribution are listed in the [dataset catalog](docs/dataset_catalog.md).

Granada patient records, private checkpoints, and generated patient reports are excluded from Git. Access follows the source dataset's agreement. The repository's [MIT license](LICENSE) does not grant rights to third-party datasets or documents.
