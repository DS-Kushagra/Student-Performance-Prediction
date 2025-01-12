# Predicting Student Performance

## Overview
This project aims to predict a student's **math score** based on various input features. The dataset used in this project is titled **"Student Performance"** and includes both numerical and categorical features. The prediction pipeline incorporates data preprocessing, model training, hyperparameter tuning, and deployment via a Flask web application.

---

## Dataset Details
- **Target Feature (Output)**:
  - `math_score`

- **Input Features**:
  1. **Numerical Features** (2):
     - `reading_score`
     - `writing_score`
  2. **Categorical Features** (5):
     - `gender`
     - `race_ethnicity`
     - `parental_level_of_education`
     - `lunch`
     - `test_preparation_course`

---

## Project Components

### 1. **Core Files**
#### a) `predict_pipeline.py`
This script defines the classes for making predictions:
- **`PredictPipeline`**:
  - Loads the trained model and preprocessor from the `artifacts/` directory.
  - Scales the input features using the preprocessor and makes predictions using the model.
- **`CustomData`**:
  - Accepts user input via web forms.
  - Converts the input into a pandas DataFrame suitable for prediction.

#### b) `exception.py`
This script handles custom exceptions:
- Logs detailed error messages, including the script name and line number where the error occurred.
- Provides consistent error handling throughout the project.

#### c) `logger.py`
Initializes a logging system to track application activity and errors:
- Logs are stored in the `logs/` directory.
- Each log file is timestamped for easy tracking.

#### d) `utils.py`
Utility functions for:
- Saving and loading serialized objects (e.g., model, preprocessor) using `dill`.
- Evaluating models with `GridSearchCV` and reporting performance metrics such as R² scores.

#### e) `app.py`
A Flask-based web application that serves as the user interface:
- **Route `/`**: Renders the homepage.
- **Route `/predictdata/`**:
  - Accepts user input from a form.
  - Processes the input through the pipeline and displays predictions.

---

### 2. **Setup and Installation**

#### `setup.py`
This file is used for packaging the project.
- **`get_requires`**: Reads the `requirements.txt` file to install necessary dependencies.

#### Installation Steps:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```

---

### 3. **Folder Structure**
```
ML-Project/
├── app.py               # Flask application
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   ├── utils.py
│   ├── exception.py
│   ├── logger.py
├── artifacts/
│   ├── model.pkl        # Serialized model
│   ├── preprocessor.pkl # Serialized preprocessor
│   ├── data.csv # Raw data
│   ├── test.csv # Splitted data for testing & model prediction
│   ├── train.csv # Used for training the ML Model
├── templates/
│   ├── index.html       # Homepage
│   ├── home.html        # Form for predictions
├── logs/
│   ├── *.log            # Log files
├── requirements.txt     # Dependencies
├── setup.py             # Setup script
```

---

## Features of the Application
1. **Data Preprocessing**:
   - Encodes categorical variables (e.g., `gender`, `lunch`).
   - Standardizes numerical variables (`reading_score`, `writing_score`).
2. **Model Training and Evaluation**:
   - Utilizes `GridSearchCV` for hyperparameter tuning.
   - Evaluates models using metrics such as R² scores on training and test datasets.
3. **Web Application**:
   - Accepts user inputs via a web form.
   - Displays predicted math scores in real time.

---

## Example Usage
### Input Form:
The web form accepts the following inputs:
- **Gender**: e.g., Male or Female.
- **Race/Ethnicity**: e.g., Group A, Group B, etc.
- **Parental Level of Education**: e.g., Bachelor's degree, Master's degree, etc.
- **Lunch**: e.g., Standard or Free/Reduced.
- **Test Preparation Course**: e.g., Completed or None.
- **Reading Score**: A numeric value (e.g., 72).
- **Writing Score**: A numeric value (e.g., 74).

### Example Input:
```json
{
  "gender": "Male",
  "race_ethnicity": "Group B",
  "parental_level_of_education": "Bachelor's degree",
  "lunch": "Standard",
  "test_preparation_course": "None",
  "reading_score": 72,
  "writing_score": 74
}
```

### Example Output:
```
Predicted Math Score: 78
```
### Image Explanation:
index.html:
![image](https://github.com/user-attachments/assets/62c1f312-6c17-4e63-a87e-083ef40d6a61)

Prediction Page:
![image](https://github.com/user-attachments/assets/bc90720b-ac78-4043-b40a-2bbd36174c8c)

---

## Key Highlights
- **Custom Exception Handling**: Provides detailed error messages for debugging.
- **Modular Code Design**: Separation of concerns between pipeline components, utilities, and web application.
- **Interactive Web Interface**: Simplifies user interaction with the prediction model.

---

## Future Enhancements
- Include more features for better prediction accuracy.
- Optimize model performance through advanced algorithms.
- Deploy the application to a cloud platform (e.g., AWS, Heroku, Azure).
- Enhance the UI with modern design frameworks.
- Add an API for programmatic access to predictions.

---

## Contact
**Author**: Kushagra  
**Email**: [kushagraagrawal128@gmail.com](mailto:kushagraagrawal128@gmail.com)

