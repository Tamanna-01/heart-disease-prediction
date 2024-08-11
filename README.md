# Heart Disease Prediction

This project involves building a machine learning model to predict the likelihood of heart disease based on various patient attributes. The model is implemented using Random Forest and is deployed via a Flask web application.

## Project Overview

The goal of this project is to create a predictive model that can help in the early detection of heart disease by analyzing input medical data. The model is trained on a dataset containing various patient features and their corresponding diagnosis results.

## Features

- **Model Training**: Utilizes Random Forest to predict the presence of heart disease.
- **Web Application**: A user-friendly interface for inputting patient data and receiving predictions.
- **Visualizations**: Includes plots for data distribution and model performance evaluation.

## Technologies Used

- **Python**: Programming language used for model training and web application development.
- **Flask**: Web framework for building the web application.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical computations.
- **Matplotlib** and **Seaborn**: Libraries for data visualization.
- **Joblib**: For saving and loading the trained model and scaler.

## Installation

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2.  **Install Dependencies**: Create a virtual environment and install the required packages.
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    
3.  **Prepare the Dataset**: Place the dataset file (`dataset.csv`) in the `data` directory.
    
4.  **Train the Model**: Run the model training script to train and save the model.
    
    ```bash
    python train_model.py
    ```
    
5.  **Run the Web Application**: Start the Flask web application to interact with the model.
    
    ```bash
    python web_app/app.py
    ```
    
6.  **Access the Web Application**: Open a web browser and go to `http://127.0.0.1:5000` to use the application.
    

Project Structure
-----------------

```
heart_disease_project/
│
├── data/
│   ├── dataset.csv               # Dataset file
│
├── models/
│   ├── heart_disease_model.pkl   # Trained machine learning model
│   ├── scaler.pkl                # Scaler for feature scaling
│
├── web_app/
│   ├── app.py                    # Flask application code
│   ├── templates/
│   │   ├── index.html            # HTML template for the web interface
│   ├── static/
│       ├── styles.css            # CSS for styling
│
├── train_model.py                # Script for training the model
├── requirements.txt              # List of Python dependencies
├── README.md                     # Project overview and setup instructions
└── .gitignore                    # Git ignore file
```

Dependencies
------------

*   `Flask`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `joblib`

Acknowledgements
----------------

*   The dataset used is from Kaggle.
*   The Random Forest algorithm and other machine learning techniques are implemented using Scikit-learn.
