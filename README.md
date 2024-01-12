# nstemi-ensemble

This is the repository containing sample codes used for the paper entitled "Data Analytics Approach for Short- and Long-Term Mortality Prediction Following Acute Non-ST-Elevation Myocardial Infarction (NSTEMI) and Unstable Angina (UA) in Asians"

## Usage
1. Load your data by replacing `"train_data.csv"` and `"test_data.csv"` with your own data files.
2. Modify feature selection as per your dataset in the section where `allMeta_list` is created.
3. Run the script section by section, following the STEP comments in the code.
4. After running the full script, your models will be saved, and a performance summary will be outputted.

## Dependencies
The script depends on the following R packages:
- `mlbench`
- `caret`
- `PROC`
- `pROC`
- `e1071`
- `tibble`
- `forcats`
- `doParallel`
- `readxl`

Ensure these packages are installed and loaded as shown in the script.

## Data
Your dataset should include a mixture of categorical and continuous features, along with an outcome variable.

## Contributing
Contributions to the script are welcome. Please fork the repository, make changes, and submit a pull request.

## Contact
Please email to [sorayya.um.edu.my](mailto:sorayya.um.edu.my) for any further inquiries.

## Full Script Description
The R script follows these steps for the machine learning workflow:

### STEP 1 - Load the Data
- Data is loaded from `.csv` files for training and testing.
- Features metadata file is optional but recommended for ease of use. An example of how the files look can be found in the `features_metadata.xlsx` file.

### STEP 2 - Data Preprocessing
- Categorical non-binary features are changed to factors.
- Binary features are converted to 0 and 1.
- Continuous and ordinal features are standardized.

### STEP 3 - Data Conversion
- Data is converted to matrices for model training.

### STEP 4 - Individual Model Building
- Individual models are built using various algorithms (SVM, RF, XGBoost, Naive Bayes).
- Cross-validation is performed for model evaluation.

### STEP 5 - Ensemble Stacking Model Building
- An ensemble model is built using stacking.
- The meta learner used in stacking is generalized linear model (GLM).

### STEP 6 - Model Evaluation
- Model performance is evaluated using ROC and AUC.
- Precision-Recall curves and confusion matrices are generated.
- Results are saved to `.csv` files for further analysis.
