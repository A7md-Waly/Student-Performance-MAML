import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Set the Streamlit server port dynamically
PORT = int(os.environ.get("PORT", 8000))


class StudentPerformanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

@st.cache_resource
def load_model():
    model = StudentPerformanceModel()
    state_dict = torch.load("student_performance_maml.pth")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

y_min, y_max = 0, 60

def predict_final_score(features):
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        normalized_prediction = model(features_tensor).item()
    prediction = normalized_prediction * (y_max - y_min) + y_min
    return int(np.ceil(prediction))

st.title("Student Performance Prediction App")
st.write("Enter the student details to predict the final exam score.")

feature1 = st.number_input("MidTerm Mark", value=0.0)
feature2 = st.number_input("Practicle Mark", value=0.0)
feature3 = st.number_input("Quizzed Mark", value=0.0)
feature4 = st.number_input("Attendence Matk", value=0.0)

if st.button("Predict Score"):
    user_features = [feature1, feature2, feature3, feature4]
    predicted_score = predict_final_score(user_features)
    st.success(f"Predicted Final Exam Score: {predicted_score}")


