import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv(r'C:\Users\moham\OneDrive\Bureau\projects\classification\logistic regression\heart_failure_clinical_records_dataset.csv')

# Convert categorical data to numerical using dummy variables
categorical_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True,dtype=int)

# Separate features (x) and target (y)
x = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Initialize and fit the Random Forest classifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)

# Make predictions
y_pred = rf.predict(x_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

# Calculate and print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cnf_matrix)
import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)