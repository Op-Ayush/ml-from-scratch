# ML From Scratch

ML algorithms built from scratch using only NumPy. No sklearn models used.

## Models Built

| Model | Dataset | Score |
|-------|---------|-------|
| Linear Regression | Iris | R²=0.93 |
| KNN | Iris | Accuracy=95.33% |
| Logistic Regression | Breast Cancer | Accuracy=98.77% |
| Decision Tree | Iris | 100% (train) | Gini impurity, recursive splitting |

## What I learned
- Derived gradient descent mathematically before coding
- Implemented MSE and cross-entropy loss from scratch
- Understood why logistic regression needs sigmoid
- KNN needs no training — just distance calculation

## How to Run
```bash
python iris_project.py
python breast_cancer_logistic.py
```
