# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Read the dataset
df = pd.read_csv('loan.csv')

# Fill in the missing values
cat_data = []
num_data = []
for i,c in enumerate(df.dtypes):
  if c==object:
    cat_data.append(df.iloc[:,i])
  else:
    num_data.append(df.iloc[:,i])
cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

# For categorical variables, we will replace missing values with the values that occur most frequently.
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
print(cat_data.isnull().sum().any())

# For numerical variables, we will replace missing values with the previous value in the same column.
num_data.fillna(method='bfill', inplace=True)
print(num_data.isnull().sum().any())

# Transform the target column 'Loan_Status' into 0 and 1
target_value = {'Y': 1, 'N': 0}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_value)
print(target)

# Replace categorical values with numerical values 0, 1, 2...
le = LabelEncoder()
for i in cat_data:
  cat_data[i] = le.fit_transform(cat_data[i])
print(cat_data)

# Remove loan_id column
cat_data.drop('Loan_ID', axis=1, inplace=True)

# Concatenate cat_data and num_data and specify the target column
X = pd.concat([cat_data,num_data], axis=1)
y = target
print(y)

# Divide the database into a test set and a training set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X,y):
  X_train, X_test = X.iloc[train], X.iloc[test]
  y_train, y_test = y.iloc[train], y.iloc[test]

print('X_train size: ', X_train.shape)
print('X_test size: ', X_test.shape)
print('y_train size: ', y_train.shape)
print('y_test size: ', y_test.shape)

# We will apply three algorithms: Logistic Regression, KNN, and Decision Tree
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}

# The accuracy function
def accu(y_true, y_pred, retu=False):
  acc = accuracy_score(y_true,y_pred)
  if retu:
    return acc
  else:
    print(f'la precision du modèle est: {acc}')

# Model application function
def train_test_eval(models, X_train, y_train, X_test, y_test):
  for name, model in models.items():
    print(name, ':')
    model.fit(X_train, y_train)
    accu(y_test, model.predict(X_test))
    print('-' * 30)

train_test_eval(models, X_train, y_train, X_test, y_test)

X_2 = X[['Gender', 'Married', 'Education', 'CoapplicantIncome', 'Credit_History']]

# Divide the database into a new test set and a training set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X_2,y):
  X_train, X_test = X_2.iloc[train], X_2.iloc[test]
  y_train, y_test = y.iloc[train], y.iloc[test]

print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)

train_test_eval(models, X_train, y_train, X_test, y_test)

# Apply logistic regression to our database
classifier = LogisticRegression()
classifier.fit(X_2, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl', 'wb'))
