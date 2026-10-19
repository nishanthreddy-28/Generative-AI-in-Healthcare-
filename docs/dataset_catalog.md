# Dataset catalog

## Pima diabetes

Location: `projects/pima_diabetes/diabetes.csv`. 768 records, eight predictors and binary `Outcome` (500 negative, 268 positive). Task: cross-sectional classification. No exact duplicate predictor profiles were found in the October 19 review. Physiological zeros are handled as missing by the existing preprocessing. Current source metadata does not replace full provenance/usage documentation.

## Early-stage diabetes

Location: `projects/early_stage_diabetes/pre_diabetic.csv`. 520 records, 16 predictors, binary `class` (320 Positive, 200 Negative). Schema matches [UCI Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction), DOI 10.24432/C5VG8H, CC BY 4.0. Exact-file provenance still needs documentation. The former folder name did not establish a clinical prediabetes target. Repeated profiles require group-aware evaluation.

## T1DiabetesGranada

Location: `projects/granada_t1d/`; patient data is local-only. Longitudinal type 1 diabetes sensor, biochemical, demographic and diagnostic tables belong to one dataset project. See the [primary dataset paper](https://www.nature.com/articles/s41597-023-02737-4) for permission-based access. Corrected-release metrics and source methods are distinct from the legacy multimodal artifacts and historical manuscript. No patient records are included in this publication.
