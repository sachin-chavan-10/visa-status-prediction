# AI Enabled Visa Status & Processing Time Estimator

# Milestone 1: Data collection and preprocessing.

## Objective
- To prepare a clean and structured dataset for predicting visa case status and processing time using machine learning.

## Data Loading
- Loaded raw visa data from datasets/raw_data.csv
- Performed initial inspection (head, info, null checks)

## Data Cleaning & Preparation
- Handled missing values:
- Categorical → filled with "Unknown"
- Numerical → filled with median values
- Fixed data types for numerical and date columns
- Standardized categorical labels

## Encoded categorical and binary data
- Lable encoding for binary categorical values(has_job_experience, requires_job_training,full_time_positon)
- One hot encoding for categorical feaures (continent, region of employment, and education level)

## Date Handling
- Generated synthetic application_date (2016–2024)
- Created decision_date using processing time
- Extracted application_year and application_month
- Dropped raw date fields after extraction

## Processing Time Generation
- Generated realistic processing_time_days using rule-based logic
- Factors considered: job type, wage, company size, case status

## Feature Engineering
- Created company_age from year of establishment
- Normalized wages to annual format
- Removed unused or redundant columns

## Target Preparation
- Standardized visa outcomes:
- Certified → Approved 

#### Identified targets:
- Classification: case_status
- Regression: processing_time_days

## Final Dataset
- Selected final feature set
- Exported cleaned dataset as:Final_Cleaned.csv

# Milestone 2: Exploratory Data Analysis (EDA) & Feature Engineering
## Objective

- To analyze the cleaned dataset, extract insights, and prepare meaningful features for machine learning models.

## Steps Completed

- Performed Exploratory Data Analysis (EDA) using visualizations and correlation analysis.

- Analyzed relationships between input features and target variables:

- 1. case_status

- 2. processing_time_days

- Generated correlation heatmaps, including filtered correlations (|corr| > 0.1).

- Identified weak linear correlations and justified feature engineering for non-linear models.

## Feature Engineering

- Created wage bands (low, medium, high, very high).

- Categorized company size (small, medium, large, enterprise).

- Generated seasonal features and peak-season indicator.

- Added region-level processing indicator.

- Ensured all engineered features are binary (0/1).

## Outcome

- Final dataset contains fully numeric, model-ready features.

- Dataset is prepared for classification and regression modeling in the next milestone.