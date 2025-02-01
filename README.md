# Diabetes Prediction Using Regression Models
This project aims to predict diabetes using **Linear Regression** and **Logistic Regression** models. The dataset contains **8 features** along with corresponding **binary labels** indicating the presence of diabetes. The models are trained, validated, and tested to achieve optimal performance.

## Table of Contents
- [Dataset Details](#dataset-details)
- [Data Processing](#data-processing)
- [Model Selection](#model-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Final Results](#final-results)
- [Conclusion](#conclusion)

## Dataset Details
The dataset consists of **619 examples** split into training, validation, and test sets as follows:

| Dataset Name       | Number of Examples |
|--------------------|-------------------|
| Training Set      | 537               |
| Validation Set    | 115               |
| Test Set         | 115               |

## Data Processing
### Pre-processing
- Checked for **NaN values** (none found).
- Converted data types where necessary.
- Applied **feature scaling** using Min-Max normalization.

### Feature Scaling
$$x' = \frac{x - min(x)}{max(x) - min(x)}$$

## Model Selection
**Regression Model:**
- Different polynomial degrees were tested (1, 2, 4, 8, 10, 12).
- **2nd-degree polynomial** was selected based on error analysis.
- Optimal **regularization parameter (λ) = 0.01**.

**Classification Model:**
- Polynomial degrees were evaluated similarly.
- **2nd-degree polynomial** was selected.
- Optimal **λ = 0**.

## Model Training and Evaluation
### Regression Model Results
| Metric  | Training | Validation | Test |
|---------|--------------|----------------|-----------|
| Cost Errors | 0.0002106 | 4.569e-06 | 0.0004257 |
| Accuracy (%) | 79.09 | 80.01 | 81.03 |

- Learning rate (α): **0.01**
- Regularization (λ): **0.01**
- Training Iterations: **5,000**

### Classification Model Results
| Metric  | Training | Validation | Test |
|---------|--------------|----------------|-----------|
| Cost Errors | 0.4497 | 0.4621 | 0.4637 |
| Accuracy (%) | 78.03 | 80.87 | 79.31 |

- Learning rate (α): **0.1**
- Regularization (λ): **0**
- Training Iterations: **10,000**

## Final Results
| Model Type       | Training Accuracy (%) | Validation Accuracy (%) | Test Accuracy (%) |
|------------------|----------------------|----------------------|------------------|
| Regression      | 79.09 | 80.01 | 81.03 |
| Classification  | 78.03 | 80.87 | 79.31 |
| Improved Classification | 93.05 | 92.98 | 95.65 |

## Conclusion
- The models successfully classified diabetes based on given features.
- The **classification model achieved 95.65% accuracy on the test set** after tuning parameters.
- Performance can be further improved by increasing **data size** and **adding more relevant features**.