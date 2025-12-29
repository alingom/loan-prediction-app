### Problem Definition

The loan prediction problem consists of building a supervised machine learning model that predicts whether a loan application should be **approved or rejected** based on an applicant's demographic, financial, and loan-related attributes.
Formally, it is a **binary classification problem**, where the target variable indicates loan approval status (Loan_Status: Yes/No).


### Motivation (African Context)

In many African countries, financial institutions face challenges such as:

- Limited access to credit for individuals and small businesses

- High default risk due to informal income sources

- Manual and subjective loan approval processes

An automated loan prediction system helps:

- Improve financial inclusion by enabling fairer credit decisions

- Reduce human bias and operational costs

- Support microfinance institutions and banks in making data-driven lending decisions, especially in underbanked populations

Such models are particularly valuable in African contexts where credit history data may be scarce and alternative features must be leveraged effectively.


### Dataset Description

The dataset contains information about loan applicants and their loan approval outcomes. It includes **demographic, financial, and loan-specific features.**

Key characteristics:

- **Source:** Kaggle – Loan Prediction Problem Dataset

- **Target variable:** Loan_Status (Approved / Not Approved)

- **Feature types:**

    - Demographic: Gender, Marital Status, Education, Dependents

    - Financial: ApplicantIncome, CoapplicantIncome, Credit_History

    - Loan-related: LoanAmount, Loan_Amount_Term, Property_Area

The dataset includes both **categorical and numerical variables**, requiring preprocessing steps such as encoding and missing value handling.


### Proposed Method

The proposed approach follows a standard machine learning pipeline:

1. **Data preprocessing**

    - Handling missing values

    - Encoding categorical variables

    - Feature scaling (if required)

2. **Exploratory Data Analysis (EDA)**

    - Understanding feature distributions

    - Identifying relationships between features and loan approval

3. **Model training**

    - Applying classification algorithms such as Logistic Regression, Decision Trees, or Random Forests

    - Training on historical loan data

4. **Model evaluation**

    - Using metrics such as accuracy, precision, recall, and confusion matrix

5. **Fairness Analysis and Tests**

    - Analyzing model predictions across sensitive attributes such as gender and marital status and by identifying potential biases in approval outcomes 

The final model is used to predict loan approval outcomes for new applicants based on learned patterns in the data.


### Project Structure
This project has three major parts :
1. model.py - This contains code for our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter loan detail and displays the predicted loan.


### Running the project
1. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

Or Run streamlit_app.py using below command to start Streamlit App

```
streamlit run streamlit_app.py
```

3. Navigate to URL http://localhost:5000 for Flask API or http://localhost:8501 for Streamlit App
