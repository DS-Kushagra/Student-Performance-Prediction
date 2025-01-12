from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Validate form data
            required_fields = ['gender', 'ethnicity', 'parental_level_of_education', 'lunch', 
                               'test_preparation_course', 'reading_score', 'writing_score']
            if any(request.form.get(field) is None for field in required_fields):
                return render_template('home.html', error="All fields are required!")
            
            # Extract data from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score')),
            )

            # Prepare data for prediction
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Input DataFrame: \n{pred_df}")
            
            # Make predictions
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction Results: {results}")
            
            return render_template('home.html', results=results[0])
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('home.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
