# ML From Scratch

Machine Learning algorithms implemented from scratch using only **NumPy**. No pre-built machine learning models from scikit-learn were used—only datasets for testing.

## Models Implemented

| Model | Dataset | Performance |
|--------|---------|-------------|
| Linear Regression | Iris | **R² = 0.93** |
| K-Nearest Neighbors (KNN) | Iris | **Accuracy = 95.33%** |
| Logistic Regression | Breast Cancer | **Accuracy = 98.77%** |
| Decision Tree | Iris | **100% Training Accuracy** |
| Random Forest | Iris | **98.67% Training Accuracy** |

## Features

- Linear Regression using Gradient Descent
- Logistic Regression with Sigmoid Activation
- Binary Cross-Entropy Loss
- KNN using Euclidean Distance
- Decision Tree using Gini Impurity and Recursive Splitting
- Random Forest using Bootstrap Sampling and Majority Voting
- Feature Normalization using NumPy

## What I Learned

- Derived gradient descent mathematically before implementing it.
- Built Linear and Logistic Regression without ML libraries.
- Implemented Mean Squared Error (MSE) and Binary Cross-Entropy loss from scratch.
- Understood why Logistic Regression requires the sigmoid function.
- Learned that KNN is a lazy learning algorithm and performs prediction using distance calculations instead of training.
- Implemented Decision Trees using Gini Impurity and recursive splitting.
- Built a Random Forest by combining multiple Decision Trees with bootstrap sampling and majority voting.

## Tech Stack

- Python
- NumPy
- SciPy (for mode in KNN)
- Matplotlib (for visualization, where applicable)
- scikit-learn (datasets only)

## How to Run

```bash
python iris_project.py
python breast_cancer_logistic.py
python decision_tree.py
python random_forest.py
```
