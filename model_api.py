# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("model_api")

# Create input/output pydantic models
input_model = create_model("model_api_input", **{'age': 51.0, 'sex': 1.0, 'cp': 2.0, 'trestbps': 100.0, 'chol': 222.0, 'fbs': 0.0, 'restecg': 0.0, 'thalach': 143.0, 'exang': 1.0, 'oldpeak': 1.2000000476837158, 'slope': 1.0, 'ca': 0.0, 'thal': 0.0})
output_model = create_model("model_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
