# Experiment 3: Git-Based Version Control and Collaborative ML Workflows

## 1. Objective

To implement Git-based version control for the Global Superstore MLOps project and understand branches, commits, merging, rollback, revert, and merge conflict resolution.

## 2. Git Repository

- Git repository initialized for the project.
- Project files tracked using Git.
- `.gitignore` configured to exclude virtual environments, Python cache files, temporary files, and MLflow tracking files.

## 3. Branch Management

The following branches were used:

- `master` – main project branch.
- `development` – development and testing changes.
- `rollback-demo` – separate branch for demonstrating rollback.

## 4. Commits and Merging

Multiple commits were created to track project changes. The development branch was merged into the master branch.

## 5. Merge Conflict Resolution

A merge conflict was intentionally created by modifying the same file differently in the master and development branches.

The conflict was resolved manually, and the resolved changes were committed.

## 6. Rollback and Revert

- `git reset --hard HEAD~1` was demonstrated on the separate `rollback-demo` branch to move back one commit.
- `git revert HEAD` was demonstrated on the master branch to reverse a commit while preserving Git history.

## 7. Project Traceability

Git tracks the following project components:

| Component | Project file |
|---|---|
| Raw dataset | `data/raw/global_superstore.csv` |
| Preprocessing code | `src/preprocess_data.py` |
| Processed dataset | `data/processed/v1/global_superstore_clean.csv` |
| Dataset version information | `data/processed/v1/manifest.json` |
| Trained model | `models/lab2/v1/profitability_model.joblib` |
| Model metadata | `models/lab2/v1/model_metadata.json` |
| Evaluation results | `reports/lab2_model_comparison.csv` |

## 8. Result

Git-based version control was successfully demonstrated using the Global Superstore MLOps project. Repository organization, branching, commits, merging, merge conflict resolution, rollback, revert, and project artifact traceability were implemented.