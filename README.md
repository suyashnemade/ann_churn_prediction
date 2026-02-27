# Customer Churn Prediction App

This project is a machine learning application designed to predict customer churn using an Artificial Neural Network (ANN). Built with TensorFlow, Keras, and Streamlit, the app provides a user-friendly web interface where users can input customer details like geography, age, and balance to get real-time churn probability.

## Key Features
- **ANN Model**: Trained using TensorFlow/Keras on a dataset of 10,000 customers.
- **Interactive UI**: A Streamlit dashboard for easy input and prediction.
- **Preprocessing**: Includes standardization (`StandardScaler`), One-Hot Encoding (Geography), and Label Encoding (Gender) ensuring inputs match the trained model's requirements.

## How to Run:
1. Ensure `requirements.txt` dependencies are installed (e.g., `pip install -r requirements.txt`).
2. Navigate to the `notebooks` directory.
3. Run the Streamlit app: `streamlit run app.py`