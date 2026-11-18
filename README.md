# Student Depression Classification

A Machine Learning project focused on exploring student-related data, identifying patterns, and developing classification models to predict student depression labels.

## Project Overview

Student mental health is an important topic in academic environments. This project explores a dataset containing demographic, academic, social, and health-related information about students.

The main objective is to analyze the dataset, identify relevant patterns, and develop machine learning classification models to predict the provided depression labels.

The project follows a complete machine learning workflow, from data exploration and preprocessing to model development, evaluation, and comparison.

## Objectives

- Explore and understand the dataset.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Analyze relationships between student characteristics and depression labels.
- Build classification models using Logistic Regression and Decision Tree.
- Evaluate model performance using appropriate classification metrics.
- Compare model performance under different feature configurations.
- Investigate the influence of `Health_condition` on model predictions.
- Identify potential data limitations affecting model reliability.

## Dataset

The project uses the **Student Depression Detection Dataset**, available on Kaggle.

- **Source:** [Student Depression Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset/data)
- **Observations:** 4,464
- **Features:** 12 input features
- **Target variable:** `Depression`

### Dataset Features

| Feature | Description |
|---|---|
| Gender | Student gender |
| Age | Student age |
| Family_background | Family background |
| Relationship_status | Relationship status |
| Marital_status | Marital status |
| Living_style | Student living arrangement |
| Current_semester | Current academic semester |
| CGPA | Cumulative Grade Point Average |
| friend_circle | Information about the student's friend circle |
| Health_condition | Health condition category |
| Cost_without_study | Cost excluding study expenses |
| Home_division | Home division |
| Depression | Target variable |

### Target Distribution

| Class | Number of Students | Percentage |
|---|---:|---:|
| No | 3,263 | 73.1% |
| Yes | 1,201 | 26.9% |

The target variable presents a moderate class imbalance, which makes it important to evaluate models using more than accuracy alone.

### Dataset Access

The dataset is not included in this repository.

To use the project:

1. Download the dataset from the Kaggle source.
2. Extract the CSV file if necessary.
3. Place `Student_Depression_Dataset.csv` inside `data/raw/`.

Please refer to `data/README.md` for detailed instructions.

## Project Structure

```text
student-depression-detection/
│
├── data/
│   ├── raw/
│   └── README.md
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_decision_tree.ipynb
│   └── 06_model_comparison.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies and Libraries

- **Python:** Main programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical visualization.
- **Scikit-learn:** Machine learning and model evaluation.
- **Jupyter Notebook:** Interactive development environment.

## Project Workflow

### 1. Data Exploration

- Load and inspect the dataset.
- Examine dataset dimensions and data types.
- Identify numerical and categorical variables.
- Analyze descriptive statistics.
- Check missing values and duplicate records.
- Examine the target variable distribution.

### 2. Data Cleaning

- Check data quality and consistency.
- Handle duplicate records.
- Investigate missing values.
- Inspect potential outliers.
- Prepare the dataset for analysis and modeling.

### 3. Exploratory Data Analysis (EDA)

- Analyze numerical feature distributions.
- Explore categorical feature distributions.
- Study relationships between features and the target variable.
- Investigate correlations and patterns.
- Visualize important findings.

### 4. Logistic Regression

A Logistic Regression classifier was developed using a Scikit-learn preprocessing pipeline.

The workflow includes:

- Stratified train-test splitting.
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Logistic Regression with `max_iter=1000`.
- Cross-validation.
- Evaluation using classification metrics.

Additional experiments were conducted to investigate the influence of `Health_condition`:

- Model using all features.
- Model excluding `Health_condition`.
- Model using only `Health_condition`.

### 5. Decision Tree

A Decision Tree Classifier was developed and evaluated.

The workflow includes:

- Initial Decision Tree model.
- Hyperparameter tuning using GridSearchCV.
- Cross-validation.
- Feature importance analysis.
- Evaluation using classification metrics.

The same feature experiments were conducted to investigate the impact of `Health_condition`.

### 6. Model Comparison

The final notebook compares Logistic Regression and Decision Tree models.

The comparison considers:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation results
- Feature importance
- The impact of removing `Health_condition`

## Model Evaluation Results

### Logistic Regression

| Experiment | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|---|---:|---:|---:|---:|
| All features | 97.09% | 96% | 93% | 94% |
| Without Health_condition | 73.12% | 50% | 20% | 29% |
| Health_condition only | 89.70% | 75% | 93% | 83% |

### Decision Tree

| Experiment | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|---|---:|---:|---:|---:|
| Tuned – All features | 99.10% | 98.33% | 98.33% | 98.33% |
| Tuned – Without Health_condition | 68.20% | 40.18% | 37.50% | 38.79% |
| Tuned – Health_condition only | 89.70% | 74.67% | 93.33% | 82.96% |

*Note: These results reflect experiments on the available dataset and evaluation setup. They do not establish clinical accuracy or real-world generalization.*

## Key Findings

- The dataset contains 4,464 observations and 12 input features.
- No missing values or duplicate records were identified during the initial dataset inspection.
- The target variable is moderately imbalanced.
- `Health_condition` has a very strong relationship with the target variable.
- Removing `Health_condition` leads to a substantial decrease in model performance.
- Decision Tree achieves higher evaluation scores than Logistic Regression when all features are included.
- Using `Health_condition` alone also produces relatively high classification performance.

### Investigating Potential Data Leakage

The strong relationship between `Health_condition` and `Depression` raises concerns about the possibility that this feature may contain information closely related to the target-label generation process.

The experiments show that:

- Some `Health_condition` categories are strongly associated with specific target classes.
- Models perform significantly worse when this feature is removed.
- The feature alone provides substantial predictive information.

However, these findings do not prove data leakage. The dataset documentation does not sufficiently explain how the depression labels were assigned.

Further investigation using independently collected data and a clearly documented labeling methodology would be necessary to assess model reliability.

## Limitations

- The methodology used to generate the target labels is not sufficiently documented.
- The relationship between `Health_condition` and `Depression` requires further investigation.
- Evaluation on a single dataset does not guarantee generalization to unseen populations.
- The dataset may not represent all student populations.
- High classification scores should not be interpreted as evidence of clinical validity.

**This project is intended for educational and portfolio purposes only. It is not a clinical diagnostic tool and must not be used to make medical decisions.**

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/hibaaouiziz/student-depression-detection.git
```

### 2. Navigate to the project directory

```bash
cd student-depression-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

Download the dataset from Kaggle and place the CSV file inside:

```text
data/raw/
```

### 7. Launch Jupyter Notebook

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `01_data_exploration.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_EDA.ipynb`
4. `04_logistic_regression.ipynb`
5. `05_decision_tree.ipynb`
6. `06_model_comparison.ipynb`

## Author

**Hiba Aouiziz**

Master's Student in Machine Learning Operations and Artificial Intelligence.

Interested in Data Science, Machine Learning, and MLOps.

**GitHub:** [hibaaouiziz](https://github.com/hibaaouiziz)

---

*Developed as a Machine Learning portfolio project.*
