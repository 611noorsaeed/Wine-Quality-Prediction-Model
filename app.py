import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the wine dataset
wine_df = pd.read_csv('winequality-red.csv')

# Create the predictor (X) and target (y) variables
X = wine_df.drop('quality', axis=1)
y = wine_df['quality'].apply(lambda yval: 1 if yval >= 7 else 0)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Display a title and input form using Streamlit
st.title("Wine Quality Prediction Model")
st.write("Please enter the wine features below to predict its quality:")

# Create input fields for each wine feature using Streamlit
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

# Get the user input and make a prediction
input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
              free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
prediction = model.predict([input_data])

# Display the prediction result using Streamlit
if st.button("Predict Quality"):
    if prediction[0] == 1:
        st.write("Good Quality Wine")
    else:
        st.write("Bad Quality Wine")
