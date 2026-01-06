**AI Enabled Visa Status & Processing Time Estimator** 

**Milestone 1: Data collection and preprocessing.**

**Objective**
To prepare a clean and structured dataset for predicting visa case status and processing time using machine learning.

**Data Loading**
Loaded raw visa data from datasets/raw_data.csv
Performed initial inspection (head, info, null checks)

**Data Cleaning & Preparation**
Handled missing values:
Categorical → filled with "Unknown"
Numerical → filled with median values
Fixed data types for numerical and date columns
Standardized categorical labels

**Date Handling**
Generated synthetic application_date (2016–2024)
Created decision_date using processing time
Extracted application_year and application_month
Dropped raw date fields after extraction

**Processing Time Generation**
Generated realistic processing_time_days using rule-based logic
Factors considered: job type, wage, company size, case status

**Feature Engineering**
Created company_age from year of establishment
Normalized wages to annual format
Removed unused or redundant columns

**Target Preparation**
Standardized visa outcomes:
Certified → Approved
Identified targets:
Classification: case_status
Regression: processing_time_days

**Final Dataset**
Selected final feature set
Exported cleaned dataset as:Final_Cleaned.csv
