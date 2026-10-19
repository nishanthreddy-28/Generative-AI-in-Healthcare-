# Local data and research artifacts

Granada data is not included in this public source distribution. Obtain authorized access through the [dataset publication and Zenodo access process](https://www.nature.com/articles/s41597-023-02737-4), following its data-use agreement. The repository code license does not grant rights to third-party datasets, documents or patient-level derivatives.

Keep the four original CSVs in `Phase_1_Data_Preparation/raw_data/`: `Patient_info.csv`, `Glucose_measurements.csv`, `Biochemical_parameters.csv`, and `Diagnostics.csv`. Keep private prepared tables, tensors, weights, scalers and indexes at the phase-relative paths expected by the code. The corrected inference release expects its original files in `Phase_5_Application/models/cgm/release_001/`; do not redistribute them without confirming applicable terms.

The original local source folder has been preserved. It contains the existing research artifacts required for model integration tests. Missing checkpoints in a public checkout are expected: code-only publication does not promise a working research model without authorized assets. Do not run historical training entrypoints against original artifacts; they can overwrite them. Use an isolated working copy and version every rebuilt dataset/model.

`docs/models/results/release_001_metrics.json` contains saved aggregate training/evaluation metadata, not raw records or checkpoint tensors. Results are not independently reproduced by copying this JSON. Notebook outputs were cleared for publication; source code was preserved. Patient-level audit evidence and generated case reports are excluded. Historical evidence links may refer to local-only files.

## Initial setup

From `projects/granada_t1d`, create a separate Python environment, install `requirements/app.txt`, and run `python -m streamlit run app.py` for the report workflow. Research requires `requirements/research.txt` plus authorized assets. Tests require `requirements/dev.txt`; full integration tests also require the research stack and private artifacts. The pinned app versions were originally recorded on Windows; a clean cross-platform installation remains to be verified. OCR additionally requires a supported local OCR backend.
