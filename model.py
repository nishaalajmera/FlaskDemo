import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset : dummy data
data = pd.read_csv('data/dummy_data.csv')
data = data.drop(['Pregnancies','SkinThickness','Insulin','DiabetesPedigreeFunction'], axis=1)
data.rename(columns={"Glucose": "glucose","BloodPressure":"bp","BMI":"bmi","Age":"age","Outcome":"outcome"}, inplace=True)
X = data.drop('outcome', axis=1) # Features
y = data['outcome'] # Labels

# Train the random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Save dummy model
pickle.dump(rf_model, open('model.pkl', 'wb'))