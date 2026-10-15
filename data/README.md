# Boston Housing Dataset Documentation

## 1. Provenance & Citation

The dataset utilized in this project is the historical **Boston Housing Dataset**, originally published in:

> Harrison, D. and Rubinfeld, D.L. *"Hedonic prices and the demand for clean air"*, Journal of Environmental Economics & Management, Vol. 5, pp. 81–102 (1978).

The benchmark was subsequent collected and distributed via the Carnegie Mellon University StatLib archive and later incorporated into educational ML libraries including Keras (`tensorflow.keras.datasets.boston_housing`).

---

## 2. Dataset Dimensions & Summary

- **Total Observations**: 506 census tracts across the Boston Standard Metropolitan Statistical Area (SMSA) in 1970.
- **Original Attributes**: 13 continuous/discrete predictors, 1 continuous target (`MEDV`).
- **Missing Values**: None (0 missing records across all 506 rows).
- **Target Variable**: `MEDV` — Median value of owner-occupied homes in thousands of US dollars (`$1,000s`).

---

## 3. Predictor Schema

| Column | Description | Data Type | Units / Range |
| :--- | :--- | :--- | :--- |
| **CRIM** | Per capita crime rate by town | Float | 0.00632 – 88.976 |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. | Float | 0.0 – 100.0% |
| **INDUS** | Proportion of non-retail business acres per town | Float | 0.46 – 27.74% |
| **CHAS** | Charles River dummy variable | Integer | 0 (No) or 1 (Yes) |
| **NOX** | Nitric oxides concentration | Float | 0.385 – 0.871 ppm |
| **RM** | Average number of rooms per dwelling | Float | 3.561 – 8.780 rooms |
| **AGE** | Proportion of owner-occupied units built prior to 1940 | Float | 2.9 – 100.0% |
| **DIS** | Weighted distances to five Boston employment centres | Float | 1.1296 – 12.1265 |
| **RAD** | Index of accessibility to radial highways | Integer | 1 – 24 |
| **TAX** | Full-value property-tax rate per $10,000 | Float | $187 – $711 |
| **PTRATIO**| Pupil-teacher ratio by town | Float | 12.6 – 22.0 |
| **B** | `1000(Bk - 0.63)^2` where `Bk` is proportion of Black residents | Float | 0.32 – 396.90 (**EXCLUDED from user interface**) |
| **LSTAT** | % lower status of the population | Float | 1.73 – 37.97% |
| **MEDV** | Median value of owner-occupied homes in $1000s | Float | $5.0 – $50.0 (Target) |

---

## 4. Ethical Feature Policy

Variable `B` is a derived racial demographic variable created by the original 1978 study authors. Because modern housing discrimination laws, ethical data science guidelines, and fair AI practices prohibit using race as a factor in property valuation:

1. Variable `B` is **not permitted** as a user-adjustable input in the web application.
2. In the 13-feature legacy Keras model from [ANN For Regression.ipynb](file:///c:/Users/Vijay/Documents/ProITBridge/Deep_Learning/Chapter8_ANN_for_Regression/ANN_for_regression/ANN%20For%20Regression.ipynb), the inference engine holds variable `B` constant at the empirical dataset mean (`356.674`), isolating it completely from user inputs.
3. In Phase 2, models will be trained exclusively on the 12 non-demographic predictors.
