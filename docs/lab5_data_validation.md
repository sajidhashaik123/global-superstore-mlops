
# Lab 5: Data Validation and Reproducible ML Pipelines

## Aim
To implement data validation and reproducible machine learning pipelines by performing schema-aware preprocessing, validating data types, detecting schema inconsistencies, missing attributes, invalid categories, and corrupted records, and building reusable, idempotent preprocessing workflows.

## Dataset
Global Superstore Dataset

- Original records: 51,290
- Original columns: 24
- Processed columns: 23
- Dataset source: data/raw/global_superstore.csv

## Tools and Technologies
- Python
- Pandas
- VS Code
- Git

## Implementation

### 1. Data Validation
Implemented checks for:
- Required and unexpected columns
- Missing values
- Data types
- Invalid categorical values
- Invalid dates
- Numeric ranges
- Duplicate records

Validation result: PASSED.

Postal Code was allowed to contain missing values because 41,296 records have missing postal codes.

### 2. Reusable Preprocessing Pipeline
Created a reusable preprocessing function that:
- Loads the Global Superstore dataset.
- Removes the Postal Code column.
- Converts Order Date and Ship Date into datetime format.
- Removes extra whitespace from text columns.
- Saves the processed dataset separately.

Output shape: 51,290 rows and 23 columns.

### 3. Idempotency Testing
Executed preprocessing twice using the same original dataset.

Result:
- First output shape: (51290, 23)
- Second output shape: (51290, 23)
- Both outputs were identical.

Idempotency test: PASSED.

### 4. Invalid Data Detection
Introduced three intentional errors in a temporary dataset:
- Invalid Ship Mode category.
- Invalid Order Date.
- Negative Sales value.

All three errors were detected successfully.

Invalid data detection test: PASSED.

## Files Created

- src/lab5_data_validation.py
- src/lab5_preprocessing_pipeline.py
- src/lab5_test_idempotency.py
- src/lab5_test_invalid_data.py
- data/processed/lab5/global_superstore_processed.csv

## Conclusion
Data validation and reproducible preprocessing were successfully implemented using the Global Superstore dataset. Schema, data types, missing values, categorical values, dates, numerical ranges, and duplicate records were checked. The preprocessing pipeline produced consistent results across repeated executions, and intentional invalid data was detected successfully.
