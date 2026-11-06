# Experiment 1: ML Project Initialization and Dataset Preparation

## 1. Aim

To implement a complete data preparation workflow using the Global Superstore dataset by analyzing its structure, data types, missing values, categorical distributions, and inconsistencies. Perform preprocessing, reproducible dataset splitting, project organization, dataset versioning, and preprocessing documentation.

## 2. Dataset Information

- **Dataset Name:** Global Superstore
- **Original File:** `global_superstore.csv`
- **Original Location:** `data/raw/`
- **Total Records:** 51,290
- **Total Columns:** 24
- **Date Range:** 01-01-2011 to 31-12-2014
- **Unique Customers:** 1,590
- **Unique Orders:** 25,035

The original dataset contains order, customer, product, geographical, sales, discount, profit, and shipping information.

## 3. Dataset Analysis

The following analysis was performed:

- Checked the number of rows and columns.
- Identified column data types.
- Examined missing values.
- Analyzed categorical column distributions.
- Checked duplicate records.
- Generated numerical summary statistics.

### Initial Dataset Findings

- Original dataset shape: 51,290 rows × 24 columns.
- Duplicate rows: 0.
- Missing values: 41,296 in the Postal Code column.
- All other columns had zero missing values.
- No blank string values were found.

## 4. Data Validation

The following validation checks were performed:

| Validation | Result |
|---|---:|
| Duplicate rows | 0 |
| Missing Postal Code values | 41,296 |
| Invalid Order Dates | 0 |
| Invalid Ship Dates | 0 |
| Ship Dates before Order Dates | 0 |
| Negative Sales values | 0 |
| Negative Quantity values | 0 |
| Negative Shipping Cost values | 0 |
| Blank string values | 0 |

The date columns were validated using the date format `%d-%m-%Y`.

Negative Profit values were preserved because they represent valid business losses.

## 5. Data Preprocessing

The following preprocessing operations were performed:

### 5.1 Handling Missing Values

The Postal Code column was removed from the processed dataset because it contained 41,296 missing values out of 51,290 records.

The original raw dataset was not modified.

### 5.2 Date Conversion

The following columns were converted into datetime format:

- Order Date
- Ship Date

The format used was `%d-%m-%Y`.

### 5.3 Text Cleaning

Leading and trailing spaces were removed from text columns using the `strip()` operation.

### 5.4 Duplicate Validation

Duplicate records were checked after preprocessing.

Result: 0 duplicate rows.

### 5.5 Final Preprocessed Dataset

- Rows: 51,290
- Columns: 23
- Missing values: 0
- Duplicate rows: 0

**Output file:** `data/processed/v1/global_superstore_clean.csv`

## 6. Reproducible Dataset Splitting

The processed dataset was divided into training and testing datasets using Scikit-learn's `train_test_split()`.

Configuration:

- Training ratio: 80%
- Testing ratio: 20%
- Random state: 42
- Shuffling: Enabled

### Split Results

| Dataset | Records |
|---|---:|
| Original processed dataset | 51,290 |
| Training dataset | 41,032 |
| Testing dataset | 10,258 |
| Total split records | 51,290 |

Overlapping Row IDs between training and testing datasets: 0.

Output files:

- `data/splits/v1/train.csv`
- `data/splits/v1/test.csv`

## 7. Dataset Versioning

The processed dataset and split files were organized under version `v1`.

A `manifest.json` file was generated to record:

- Dataset name and version.
- Raw dataset information.
- Processed dataset information.
- Preprocessing operations.
- Training and testing dataset details.
- Split ratio and random state.
- SHA-256 checksums for file integrity.

Manifest location:

`data/processed/v1/manifest.json`

SHA-256 checksums help identify changes to dataset files.

## 8. Project Repository Organization

The project was organized into the following directories:

```text
global_superstore_mlop_project/
│
├── data/
│   ├── raw/
│   │   └── global_superstore.csv
│   │
│   ├── processed/
│   │   └── v1/
│   │       ├── global_superstore_clean.csv
│   │       └── manifest.json
│   │
│   └── splits/
│       └── v1/
│           ├── train.csv
│           └── test.csv
│
├── docs/
│   └── lab1_preprocessing.md
│
├── reports/
│   ├── lab1_dataset_profile.json
│   └── lab1_dataset_profile.txt
│
├── src/
│   ├── data_loader.py
│   ├── data_analysis.py
│   ├── data_validation.py
│   ├── test_validation.py
│   ├── preprocess_data.py
│   ├── split_dataset.py
│   └── version_dataset.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 9. Reproducibility

Reproducibility was maintained through:

- A fixed random state of 42 during dataset splitting.
- Separate raw, processed, and split dataset directories.
- Dataset version identification.
- SHA-256 checksum generation.
- Documented preprocessing operations.

## 10. Conclusion

The Global Superstore dataset was successfully analyzed, validated, preprocessed, and divided into training and testing datasets. Dataset versioning and checksum generation were implemented to maintain traceability and reproducibility.

The original raw dataset was preserved, and the processed dataset was stored separately for subsequent MLOps experiments.