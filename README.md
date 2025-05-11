# SyriaTel Customer Churn Prediction

**Overview**

This repository contains a comprehensive data science project aimed at predicting customer churn for SyriaTel, a telecommunications provider. By leveraging customer account information, service plan usage metrics, and customer support interactions, the objective is to build an accurate classification model that identifies subscribers at risk of cancelling their service. Retention strategies based on these predictions can help SyriaTel reduce revenue loss and improve customer satisfaction.

---

## Business and Data Understanding

**Stakeholder:**

* SyriaTel Retention & Marketing Teams

**Business Problem:**

* Churn reduces recurring revenue and increases customer acquisition costs. Predicting churn allows targeted retention campaigns to keep high-risk customers active.

**Data Source & Description:**

* **Original Dataset:** `data/raw/syriatel_churn.csv` (3,333 rows, 21 columns)
* **Key Features:**

  * **Demographics:** `state`, `area code`, `phone number` (de-identified)
  * **Account Attributes:** `account length`, `international plan`, `voice mail plan`
  * **Usage Metrics:** Day, evening, night, and international minutes & calls
  * **Financial Metrics:** Corresponding charges (later dropped due to redundancy)
  * **Service Interactions:** `customer service calls`
  * **Target Variable:** `churn` (boolean flag)

**Initial Insights:**

* Approximately 14.5% of customers churned.
* Usage metrics were strongly correlated with charges (perfect correlation), necessitating feature reduction.
* Moderate class imbalance suggests the need for stratified sampling and specialized modeling techniques.

---

## Data Preparation & Feature Engineering

### 1. Redundant Feature Removal

* Dropped perfectly correlated charge columns (`total day charge`, etc.) to avoid multicollinearity.

### 2. Train/Test Split

* Performed an 80/20 stratified split on `churn` for balanced class representation.
* Verified that both splits maintained \~85.5% stayers and \~14.5% churners.

### 3. Cleaning & Imputation

* Converted blank strings to NaN for robustness.
* **Numeric Imputation:** Median strategy for continuous features to handle potential future missingness.
* **Categorical Imputation:** Mode strategy for binary/service-plan features.

### 4. Feature Engineering

* **Binning:** Categorized `account length` into `new`, `mid-term`, and `long-term` tenure buckets.
* **Interaction Features:** Created `intl_plan_usage` and `vm_plan_messages` to capture service flag × usage behavior.

### 5. Encoding & Scaling

* **Label Encoding:** Mapped binary flags (`yes`/`no`) to 1/0.
* **One-Hot Encoding:** Transformed multi-class categoricals (`state`, `area code`, `tenure_bin`).
* **Scaling:** Standardized all numeric features (mean=0, std=1) using `StandardScaler`.

### 6. Reproducible Pipeline

* Implemented a `ColumnTransformer` + `Pipeline` to bundle all preprocessing steps and prevent data leakage.
* Exported processed feature matrices to `data/processed/` for transparency and reuse.

---

## Modeling

We adopted an iterative modeling approach, building from simple to more complex models and tuning hyperparameters for the business goal of maximizing churn recall.

### 1. Baseline Logistic Regression

* Accuracy: 0.86, Recall (churn): 0.25, Precision: 0.53.
* Good overall accuracy but poor at catching churners.

### 2. Tuned Logistic Regression

* Optimized regularization strength (`C`) via `GridSearchCV` for recall.
* No significant improvement in recall for churners.

### 3. Decision Tree

* **Baseline:** Precision 0.90, Recall 0.67, F1 0.77, Accuracy 0.94.
* **Tuned:** Slight precision drop (0.76) for similar recall, trading interpretability vs. performance.
* **Selection:** Baseline tree chosen for its balance and ease of explanation to stakeholders.

### 4. Random Forest

* Perfect precision (1.00) but very low recall (0.12), indicating overfitting to non-churners.
* Not selected due to poor churn detection despite high accuracy.

---

## Evaluation & Business Recommendations

### Final Model Performance (Baseline Decision Tree)

* **Accuracy:** 0.94
* **Precision (Churn):** 0.90
* **Recall (Churn):** 0.67
* **F1-score (Churn):** 0.77
* **ROC-AUC:** 0.85 (computed via test set probabilities)

### Confusion Matrix

```plaintext
                Predicted Stay  Predicted Churn
Actual Stay           548                22
Actual Churn           32                65
```

* **Interpretation:** Of 97 actual churners, 65 were correctly flagged. Of 570 stayers, only 22 were incorrectly targeted.

### Business Impact

* **Retention Campaign:** Target top 10% highest-risk customers (\~67 individuals) to retain \~45 at-risk subscribers monthly.
* **Revenue Preservation:** At \$30/month/customer, salvaging 45 customers yields \$1,350/month (\$16,200/year).
* **Implementation Plan:**

  1. Deploy model in production with monthly retraining.
  2. A/B test tailored retention offers to measure lift.
  3. Monitor key metrics (actual churn vs. predicted) and refine features with new customer feedback.

---

## Repository Structure

```
├── data/
│   ├── raw/                 # Original CSV dataset
│   └── processed/           # Train/Test processed CSVs
├── notebook/               # Jupyter notebooks detailing exploration & modeling
│   ├── syriatel_churn.ipynb
├── src/                     # Python scripts for pre-processing and modeling
│   ├── preprocess.py        # Data cleaning & feature engineering
│   └── modeling.py          # Model training & evaluation
├── README.md                # Project overview and instructions
└── requirements.txt         # Python dependencies
```

---

## Getting Started

1. **Clone repository:**

   ```bash
   git clone <[[https://github.com/tedronoh-14/syriatel-churn-prediction]]>
   cd <syriatel-churn-prediction>
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Reproduce preprocessing:**

   ```bash
   python src/preprocess.py
   ```
4. **Run modeling scripts:**

   ```bash
   python src/modeling.py
   ```
5. **Explore results** in `notebooks/` or review CSV outputs in `data/processed/`.

---

## License

This project is for educational purposes and is not under a commercial license.

---
