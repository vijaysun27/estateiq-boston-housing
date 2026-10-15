# EstateIQ Model Card

## 1. Model Details

### Primary Model (Phase 2): Scikit-Learn Regression Pipeline
- **Model Name**: EstateIQ Boston Housing Scikit-Learn Pipeline
- **Model Identifier**: `boston_housing_pipeline`
- **Model Type**: Scikit-Learn Pipeline (`StandardScaler` + `GradientBoostingRegressor`)
- **Framework**: Scikit-Learn >= 1.4.0
- **Artifact Location**: `models/boston_housing_pipeline.joblib`
- **Metadata Location**: `models/model_metadata.json`
- **Training Script**: [scripts/train_model.py](file:///c:/Users/Vijay/Documents/ProITBridge/Deep_Learning/Chapter8_ANN_for_Regression/estateiq-boston-housing/scripts/train_model.py)
- **Features**: Dedicated 12-feature canonical schema (Historical `B` completely eliminated from training).

### Legacy Model (Phase 1 Foundation): Keras Sequential ANN
- **Model Name**: EstateIQ Boston Housing ANN Regressor
- **Model Identifier**: `boston_housing_ann_keras`
- **Model Type**: Artificial Neural Network (Sequential Multi-Layer Perceptron)
- **Framework**: TensorFlow Keras v2.15+
- **Artifact Location**: `models/boston_housing_model.keras`
- **Trained In**: [ANN For Regression.ipynb](file:///c:/Users/Vijay/Documents/ProITBridge/Deep_Learning/Chapter8_ANN_for_Regression/ANN_for_regression/ANN%20For%20Regression.ipynb)
- **Architecture**: `Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)` (3,521 parameters)

---

## 2. Training & Model Selection Rationale (Phase 2)

During Phase 2 training on the 404 training observations (with 102 held out for testing under fixed seed 42), five candidate pipelines were evaluated using 5-fold cross-validation on the training set:

| Model Candidate | Preprocessing | 5-Fold CV RMSE | 5-Fold CV MAE | 5-Fold CV $R^2$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyRegressor (Mean)** | None | 9.0813 | 6.6402 | -0.0103 | Baseline |
| **Linear Regression** | StandardScaler | 4.3547 | 3.1822 | 0.7595 | Evaluated |
| **Ridge Regression** | StandardScaler ($\alpha=1.0$) | 4.3501 | 3.1737 | 0.7600 | Evaluated |
| **Random Forest** | StandardScaler ($N=100, \text{depth}=6$) | 3.5476 | 2.4351 | 0.8286 | Evaluated |
| **Gradient Boosting** | StandardScaler ($N=100, \text{lr}=0.1, \text{depth}=4$) | **3.3571** | **2.2536** | **0.8460** | **Selected Final Model** |

### Held-Out Test Evaluation (102 Observations)

The selected `GradientBoostingRegressor` pipeline was fitted on all 404 training samples and evaluated on the untouched 102 held-out test samples:

| Metric | Dummy Baseline | Selected Model (Gradient Boosting) | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** ($1,000s) | 9.5319 | **3.9177** | **58.9% reduction in error** |
| **MAE** ($1,000s) | 6.7727 | **2.3815** | **64.8% reduction in error** |
| **$R^2$ Score** | -0.0003 | **0.8310** | **Explains 83.1% of variance** |

---

## 3. Dataset & Ethical Feature Policy

### Canonical Predictor Features (12 Features)
1. `CRIM`: Per capita crime rate by town
2. `ZN`: Proportion of residential land zoned for lots > 25,000 sq.ft.
3. `INDUS`: Proportion of non-retail business acres per town
4. `CHAS`: Charles River proximity dummy variable (0 or 1)
5. `NOX`: Nitric oxides concentration (parts per 10 million)
6. `RM`: Average number of rooms per dwelling
7. `AGE`: Proportion of owner-occupied units built prior to 1940
8. `DIS`: Weighted distances to five Boston employment centres
9. `RAD`: Index of accessibility to radial highways (valid levels: 1-8, 24)
10. `TAX`: Full-value property-tax rate per $10,000
11. `PTRATIO`: Pupil-teacher ratio by town
12. `LSTAT`: % lower status of the population

### Ethical Exclusion of Variable `B`
The original 1970s benchmark included a 13th variable designated as `B` (`1000(Bk - 0.63)^2`), which encoded the proportion of Black residents in each town. 

**EstateIQ Policy:**
- Variable `B` is **strictly excluded** from user inputs and client visibility.
- In Phase 2, `boston_housing_pipeline.joblib` was trained strictly on the 12 non-demographic predictors, eliminating variable `B` entirely from the weight matrix.
- The legacy Keras model isolates `B` by holding it constant at the neutral empirical mean (`356.674`).

---

## 4. Target Variable & Scale

- **Target**: `MEDV`
- **Definition**: Median value of owner-occupied homes.
- **Historical Scale**: Expressed in **thousands of US dollars ($1,000s)** as measured in 1970.
- **Display Conversion**: `prediction_dollar = raw_prediction * 1,000.0`.
- **Note**: A prediction of `24.5` represents `$24,500` (1970 benchmark value).

---

## 5. Limitations & Caveats

1. **Temporal Obsolescence**: The underlying census survey was conducted in 1970. Inflation, urbanization, zoning transformations, and demographic shifts render these figures obsolete for modern market pricing.
2. **Geographical Scope**: Limited strictly to the Boston, Massachusetts metropolitan area.
3. **Census Tract Aggregation**: Data points are aggregated at the census tract level, not individual single-family parcels.
4. **Academic Demonstration Only**: Predictions must never be used for real estate investment, mortgage underwriting, or legal appraisal.
