# Drought Prediction using ERA5 Data and Machine Learning

This project implements a machine learning pipeline for predicting drought conditions over India using ERA5 climate reanalysis data and SPEI (Standardized Precipitation Evapotranspiration Index) values.

## Project Overview

The analysis consists of 6 Jupyter notebooks that follow a complete workflow from data preparation to model evaluation and baseline comparison:

| Notebook | Description |
|----------|-------------|
| `1_DataPreparation.ipynb` | Loads ERA5 NetCDF data, merges with SPEI, clips to India shapefile, and classifies SPEI values into 5 drought categories |
| `2_EDA.ipynb` | Exploratory Data Analysis including correlation analysis, distribution plots, and baseline model training |
| `3_MLmodels.ipynb` | Creates train/validation/test datasets with z-score and robust normalization, converts gridded data to tabular format |
| `4_ModelSelection.ipynb` | Hyperparameter tuning and model selection (Random Forest, XGBoost, LightGBM, CatBoost) with performance evaluation |
| `5_SHAP_Analysis.ipynb` | SHAP (SHapley Additive exPlanations) analysis for model interpretability and feature importance visualization |
| `6_Baseline.ipynb` | Compares ML models against persistence and climatology baselines |

## Drought Classification

SPEI-1 values are classified into 5 categories:

| Class | SPEI Range | Description |
|-------|------------|-------------|
| 0 | ≥ -0.5 | Non-drought |
| 1 | -1.0 to -0.5 | Mild drought |
| 2 | -1.5 to -1.0 | Moderate drought |
| 3 | -2.0 to -1.5 | Severe drought |
| 4 | ≤ -2.0 | Extreme drought |

## Data Structure

### Input Data
- **ERA5 Monthly Data**: Located in `Data/ERA5_monthly/`
  - SPEI: NetCDF files in `Data/ERA5_monthly/SPEI/*.nc`
  - Auxiliary variables (t2m, tp, e, ro, cdir): NetCDF files in `Data/ERA5_monthly/AuxData/`
- **India Shapefile**: `Data/Ind_shapefile/`

### Output Data
- `Data/Ind_SPEI1_ERA5_monthly_5class.nc`: Processed NetCDF with classified SPEI
- `Results/`: Model predictions, metrics, and visualization outputs

## Features

The following predictor variables are used for modeling:
- `t2m`: 2-meter temperature
- `e`: Evaporation
- `ro`: Runoff
- `tp`: Total precipitation
- `cdir`: Convective drainage

### Temporal Encoding
- `month_sin`, `month_cos`: Cyclic encoding of month

### Spatial Encoding
- `sin_lat`, `cos_lat`: Cyclic encoding of latitude
- `sin_lon`, `cos_lon`: Cyclic encoding of longitude

## Models Evaluated

1. **Ensemble Model**: Weighted soft voting combining Random Forest, LightGBM, and XGBoost
2. **Random Forest**: Tuned via RandomizedSearchCV
3. **XGBoost**: Tuned via GridSearchCV with GPU acceleration
4. **LightGBM**: Tuned via RandomizedSearchCV with GPU acceleration
5. **CatBoost**: Tuned via randomized search with GPU support

## Evaluation Metrics

- Accuracy
- F1 Score (Macro)
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy

## Results

Model performance results and figures are saved in the `Results/` directory, including:
- Model performance summaries
- Confusion matrices
- SHAP analysis plots (global importance, per-class importance, seasonal analysis, spatial maps)
- Baseline comparison charts

## Requirements

Key Python libraries:
- `xarray`, `rioxarray`: NetCDF data handling
- `geopandas`: Shapefile processing
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: ML models and metrics
- `xgboost`, `lightgbm`, `catboost`: Gradient boosting models
- `shap`: Model interpretability
- `matplotlib`, `seaborn`: Visualization
- `cartopy`: Map projections (optional)