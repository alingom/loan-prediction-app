import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.write("L'application qui prédit l'accord du crédit")

# Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques du client")

def client_caract_input():
    Gender = st.sidebar.selectbox('Genre', ('Male', 'Female'))
    Married = st.sidebar.selectbox('Marié', ('Yes', 'No'))
    Education = st.sidebar.selectbox('Éducation', ('Graduate', 'Not Graduate'))
    CoapplicantIncome = st.sidebar.number_input('Salaire du conjoint', min_value=0, max_value=40000, value=2000, step=1)
    Credit_History = st.sidebar.selectbox('Historique de Crédit', (1.0, 0.0))

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


# Transformer les données d'entrée en données adaptées à notre modèle
# Importer la base de données pour obtenir les catégories
df = pd.read_csv('loan.csv')

# Séparer les données catégorisées et numériques
cat_data = []
num_data = []
for i, c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else:
        num_data.append(df.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

# Remplir les valeurs manquantes
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
num_data.fillna(method='bfill', inplace=True)

# Supprimer Loan_ID et Loan_Status
cat_data.drop('Loan_ID', axis=1, inplace=True)
cat_data.drop('Loan_Status', axis=1, inplace=True)

# Créer des LabelEncoders pour chaque variable catégorisée AVANT de les encoder
encoders = {}
for col in cat_data.columns:
    encoders[col] = LabelEncoder()
    encoders[col].fit(cat_data[col].unique())

# Encoder les variables catégorisées avec LabelEncoder
for col in cat_data.columns:
    cat_data[col] = encoders[col].transform(cat_data[col])

# Créer le dataframe complet
X = pd.concat([cat_data, num_data], axis=1)

# Préparer les données d'entrée
donnee_entree = pd.DataFrame({
    'Gender': [input_df['Gender'].values[0]],
    'Married': [input_df['Married'].values[0]],
    'Education': [input_df['Education'].values[0]],
    'CoapplicantIncome': [input_df['CoapplicantIncome'].values[0]],
    'Credit_History': [input_df['Credit_History'].values[0]]
})

# Encoder les variables catégorisées
donnee_entree['Gender'] = encoders['Gender'].transform(donnee_entree['Gender'])
donnee_entree['Married'] = encoders['Married'].transform(donnee_entree['Married'])
donnee_entree['Education'] = encoders['Education'].transform(donnee_entree['Education'])

# Sélectionner les features dans le bon ordre
donnee_entree = donnee_entree[['Gender', 'Married', 'Education', 'CoapplicantIncome', 'Credit_History']]

# Afficher les données transformées
st.subheader('Les caracteristiques transformées')
st.write(donnee_entree)


#importer le modèle
load_model = pickle.load(open('model.pkl', 'rb'))


#appliquer le modèle sur le profil d'entrée
prediction = load_model.predict(donnee_entree)
output = round(prediction[0], 2)

# Afficher le résultat
st.subheader('Résultat de la prédiction')
st.write('Loan Prediction is {}'.format(output))