# Student Success Predictor

An early-warning machine learning application that predicts whether a student is likely to **Dropout, remain Enrolled, or Graduate** based on academic, personal, and financial information.

The project combines **XGBoost**, **SHAP**, and **Streamlit** to provide predictions together with explanations of the factors influencing the model's output.

## Project Overview

The Student Success Predictor is designed as a decision-support application for identifying students who may benefit from early academic or financial support.

The Streamlit application allows users to enter student information and receive:

- Predicted student outcome
- Dropout probability
- Low, Medium, or High dropout-risk level
- Class probabilities
- SHAP-based explanation of the prediction
- What-if analysis
- Cohort-level predictions
- Model performance metrics
- Suggested support actions

## Machine Learning Model

The project uses an **XGBoost classification model** to predict three possible outcomes:

- Dropout
- Enrolled
- Graduate

The trained model is saved as:

```text
dropout_model.pkl
```

The Streamlit application loads this trained model and uses it to generate predictions.

## Explainable AI

**SHAP (SHapley Additive exPlanations)** is used to explain individual predictions.

The application shows which features have the strongest influence on a particular prediction, helping users understand why the model produced a particular result.

## Key Features

### 1. Student Prediction

Users can enter student information including:

- Admission grade
- Previous qualification grade
- First-semester academic information
- Second-semester academic information
- Age at enrollment
- Gender
- International student status
- Displacement status
- Scholarship status
- Debtor status
- Tuition-fee status

The model then predicts the most likely outcome.

### 2. Dropout Risk

The application displays the estimated probability of dropout and categorizes the risk as:

- **Low:** below 35%
- **Medium:** 35% to 60%
- **High:** 60% or above

These thresholds are used by the Streamlit application to present the dropout probability as a risk level.

### 3. Prediction Explanation

SHAP values are used to show how individual student features contribute to the selected model outcome.

### 4. What-If Analysis

The application allows users to vary selected features and observe how the predicted probabilities change while the other inputs remain unchanged.

### 5. Cohort Dashboard

A CSV file can be uploaded to score multiple students at once.

The application provides:

- Number of students scored
- Number of high-risk students
- Average dropout probability
- Predicted outcomes
- Dropout probability by age group
- Highest-risk students
- Top SHAP driver for high-risk students
- Downloadable prediction results

### 6. Model Performance

The application displays model evaluation information including:

- Test accuracy
- Macro-F1
- Training sample count
- Number of features
- Confusion matrix
- Precision
- Recall
- F1-score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Studentdropoutpred/
│
├── app.py
├── notebook.ipynb
├── dropout_model.pkl
├── data.csv
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── config.toml
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/student-dropout-predictor.git
```

Move into the project directory:

```bash
cd student-dropout-predictor
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset by Realinho, V. et al. (2021), from the UCI Machine Learning Repository.

The dataset contains student-related academic, demographic, and financial information used for predicting student outcomes.

## Responsible Use

This application is intended as **decision support**, not as a decision-maker.

A high-risk prediction should be used to identify students who may benefit from additional support rather than for penalties, restrictions, or automatic decisions.

Model predictions are based on patterns learned from the available training data and should not be interpreted as certainty or causation.

## Future Improvements

Potential improvements include:

- Testing the model on data from additional institutions
- Monitoring model performance over time
- Evaluating performance across different student groups
- Improving calibration of predicted probabilities
- Adding model monitoring and drift detection
- Deploying the application for controlled institutional use
