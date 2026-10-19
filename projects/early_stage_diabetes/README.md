# Early-stage diabetes classification

This project contains the former `pre - diabetic/` notebook and 520-row CSV. Filenames are preserved so the notebook can still read `pre_diabetic.csv` from this directory.

The 16-feature symptom schema matches [UCI Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction); confirm this exact copy against its source and provide CC BY 4.0 attribution. The Positive/Negative labels do not establish a distinct prediabetes class.

Open `PreDiabetic.ipynb` with this directory as the working directory. Dependencies are pandas and scikit-learn; use a separate environment from the other projects. No trained model is included.

## Evaluation status

The original notebook's seed-42 random split has 69 of 104 test rows with identical predictor profiles in training. The CSV has 269 duplicate rows beyond first occurrences. The notebook and dataset are preserved; a profile-grouped split and training-only selection are needed before reporting generalization.
