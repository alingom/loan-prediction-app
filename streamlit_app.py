import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.write("The application that predicts loan approval")

# Collect the entry profile of the client
st.sidebar.header("The characteristics of the customer")

def client_caract_input():
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    Education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    CoapplicantIncome = st.sidebar.number_input('Co-applicant Income', min_value=0, max_value=40000, value=2000, step=1)
    Credit_History = st.sidebar.selectbox('Credit History', (1.0, 0.0))

    data = {
    'Gender': Gender,
    'Married': Married,
    'Education': Education,
    'CoapplicantIncome': CoapplicantIncome,
    'Credit_History': Credit_History
    }

    profil_client = pd.DataFrame(data, index=[0])
    return profil_client

input_df = client_caract_input()


# Transforming input data into data suitable for our model
# Import the database to obtain the categories
df = pd.read_csv('loan.csv')

# Separate categorised and numerical data
cat_data = []
num_data = []
for i, c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else:
        num_data.append(df.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

# Fill in the missing values
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
num_data.fillna(method='bfill', inplace=True)

# Remove Loan_ID and Loan_Status
cat_data.drop('Loan_ID', axis=1, inplace=True)
cat_data.drop('Loan_Status', axis=1, inplace=True)

# Create LabelEncoders for each categorised variable BEFORE encoding them
encoders = {}
for col in cat_data.columns:
    encoders[col] = LabelEncoder()
    encoders[col].fit(cat_data[col].unique())

# Encode categorical variables with LabelEncoder
for col in cat_data.columns:
    cat_data[col] = encoders[col].transform(cat_data[col])

# Create the complete dataframe
X = pd.concat([cat_data, num_data], axis=1)

# Prepare the input data for prediction
donnee_entree = pd.DataFrame({
    'Gender': [input_df['Gender'].values[0]],
    'Married': [input_df['Married'].values[0]],
    'Education': [input_df['Education'].values[0]],
    'CoapplicantIncome': [input_df['CoapplicantIncome'].values[0]],
    'Credit_History': [input_df['Credit_History'].values[0]]
})

# Encoding categorical variables
donnee_entree['Gender'] = encoders['Gender'].transform(donnee_entree['Gender'])
donnee_entree['Married'] = encoders['Married'].transform(donnee_entree['Married'])
donnee_entree['Education'] = encoders['Education'].transform(donnee_entree['Education'])

# Select the features in the correct order
donnee_entree = donnee_entree[['Gender', 'Married', 'Education', 'CoapplicantIncome', 'Credit_History']]

# Display the transformed data
st.subheader('The transformed features of the customer')
st.write(donnee_entree)


# Import model
load_model = pickle.load(open('model.pkl', 'rb'))


# Apply the model to the input profile
prediction = load_model.predict(donnee_entree)
output = round(prediction[0], 2)

# Display the result
st.subheader('Prediction outcome')
st.write('Loan Prediction is {}'.format(output))