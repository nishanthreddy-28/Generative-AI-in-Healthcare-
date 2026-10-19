# Early-stage diabetes classification

A comparison of random forest, decision tree, and logistic regression on symptom-based diabetes data.

The dataset contains **520 records** and **16 predictors**, including age, reported symptoms, and sex. The target is `Positive` or `Negative` (320 and 200 records respectively). Its schema matches the [UCI Early Stage Diabetes Risk Prediction dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction), collected at Sylhet Diabetes Hospital in Bangladesh. Confirm the source of this exact CSV and retain the dataset's CC BY 4.0 attribution when reusing it.

This project was previously named `pre - diabetic`. The data does not contain a separate prediabetes class. Existing notebook/CSV filenames are preserved so their relative paths still work.

## Run the notebook

From `projects/early_stage_diabetes`, create a separate environment and install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas scikit-learn jupyterlab
python -m jupyterlab
```

Open `PreDiabetic.ipynb` with this folder as the working directory. It reads `pre_diabetic.csv`, encodes the categorical values, trains three classifiers, and evaluates an example input.

## Evaluation notes

The current CSV has 269 repeated rows beyond first occurrences, leaving 251 unique predictor profiles. In the notebook's seed-42 random split, **69 of 104 test rows have identical predictor profiles in training**.

This overlap can inflate the reported performance. Group identical profiles before splitting, choose models using training/validation data, and reserve a final test set. Repeated profiles do not prove that records belong to the same person; that limitation also needs to be reported.

The notebook and dataset are retained as the original experiment. No model weights are bundled, and the outputs should not be interpreted as a diagnosis.
