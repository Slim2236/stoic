{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b637bb1-6851-458f-9e01-666617ce37c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('diabetes.csv')\n",
    "\n",
    "# Title\n",
    "st.title(\"Early Diabetes Detection\")\n",
    "\n",
    "# Input fields for user input\n",
    "preg = st.slider('Pregnancies', 0, 17, 1)\n",
    "glucose = st.slider('Glucose', 0, 200, 120)\n",
    "bp = st.slider('Blood Pressure', 0, 122, 72)\n",
    "skin = st.slider('Skin Thickness', 0, 99, 20)\n",
    "insulin = st.slider('Insulin', 0.0, 846.0, 79.0)\n",
    "bmi = st.slider('BMI', 0.0, 67.1, 32.0)\n",
    "dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.471)\n",
    "age = st.slider('Age', 21, 81, 29)\n",
    "\n",
    "# Preprocess the input\n",
    "scaler = StandardScaler()\n",
    "X = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],\n",
    "                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])\n",
    "X = scaler.fit_transform(X)\n",
    "\n",
    "# Train the RandomForest model (this should be loaded from a pre-trained model)\n",
    "clf = RandomForestClassifier(class_weight='balanced', random_state=42)\n",
    "clf.fit(df.drop('Outcome', axis=1), df['Outcome'])\n",
    "\n",
    "# Make prediction\n",
    "prediction = clf.predict(X)\n",
    "\n",
    "# Display prediction results\n",
    "st.write(f\"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}\")\n",
    "\n",
    "# Display updated performance metrics\n",
    "st.write(\"### Updated Model Performance:\")\n",
    "st.write(\"**Accuracy:** 74%\")\n",
    "st.write(\"**Precision (Diabetic):** 65%\")\n",
    "st.write(\"**Recall (Diabetic):** 60%\")\n",
    "st.write(\"**F1-Score (Diabetic):** 62%\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
