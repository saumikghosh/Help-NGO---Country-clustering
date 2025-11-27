import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Let's load joblib instances over here.
with open('pipeline.joblib', 'rb') as file:
    preprocess = joblib.load(file)
with open('kmeans_model.joblib', 'rb') as file:
    model = joblib.load(file)
# Lets take input from the users
st.title("Help NGO")
st.subheader("Predicting the development category for funds required according to its socio-economic development status. Used KMeans.")
# Lets take input
gdp = st.number_input("Enter GDP (in USD)", min_value=0)
income = st.number_input("Enter Average GDP per capita (in USD)", min_value=0)
imports = st.number_input("Enter Imports (in USD)", min_value=0)
exports = st.number_input("Enter Exports (in USD)", min_value=0)
inflation_rate = st.number_input("Enter Inflation Rate (in %)", min_value=-100, max_value=100)
life_expectancy = st.number_input("Enter Life Expectancy (in years)", min_value=0)
expenditure_on_health = st.number_input("Enter Expenditure on Health (in USD)", min_value=0)
fertility_rate = st.number_input("Enter Fertility Rate (children per woman)", min_value=0)
child_mortality_rate = st.number_input("Enter Child Mortality Rate (per 1000 live births)", min_value=0)

input_list = [child_mortality_rate,exports,expenditure_on_health,imports,income,inflation_rate,life_expectancy,fertility_rate,gdp]
final_input_list = preprocess.transform([input_list])
if st.button("Predict Development Category"):
    prediction = model.predict(final_input_list)[0]
    if prediction == 0:
        st.success("The country is Underdeveloped. Requires maximum funds.")
    elif prediction == 1:
        st.success("The country is Developing. Requires moderate funds.")
    else:
        st.error("The country is Developed. Requires minimum funds.")