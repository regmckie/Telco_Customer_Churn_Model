"""
FASTAPI + GRADIO SERVING APPLICATION
====================================

FILE DESCRIPTION:
This application provides a complete serving solution for the Telco customer churn model,
with both programmatic API access and a user-friendly web interface.


ARCHITECTURE:
- FastAPI: Modern, high-performance web framework for building APIs with Python
- Gradio: User-friendly web UI for manual testing & demos
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn for telecommunications industry",
    version="1.0.0"
)

# --- HEALTH CHECK ENDPOINT ---
# Required for the AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# --- REQUEST DATA SCHEMA ---
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.

    This schema defines the exact 18 features required for churn prediction.
    For consistency, all features match the original dataset structure.
    """
    # Demographics
    gender: str  # "Male" or "Female"
    Partner: str  # "Yes" or "No" - has partner
    Dependents: str  # "Yes" or "No" - has dependents

    # Phone services
    PhoneService: str  # "Yes" or "No"
    MultipleLines: str  # "Yes", "No", or "No phone service"

    # Internet services
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service"
    OnlineBackup: str  # "Yes", "No", or "No internet service"
    DeviceProtection: str  # "Yes", "No", or "No internet service"
    TechSupport: str  # "Yes", "No", or "No internet service"
    StreamingTV: str  # "Yes", "No", or "No internet service"
    StreamingMovies: str  # "Yes", "No", or "No internet service"

    # Account information
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str  # "Electronic check", "Mailed check", etc.

    # Numeric features
    tenure: int  # Number of months with company
    MonthlyCharges: float  # Monthly charges in dollars
    TotalCharges: float  # Total charges to date

# --- MAIN PREDICTION API ENDPOINT ---
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.

    This endpoint:
    1. Receives validated customer data via the Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns prediction in JSON format

    Expected response:
    - {"prediction": "Likely to churn"} OR {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails

    :param data: Customer data with the 18 features required for churn prediction
    :return: The prediction or error as defined above in "Expected response"
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        pred_result = predict(data.dict())
        return {"prediction": pred_result}
    except Exception as e:
        # Return error details for debugging
        return {"error": str(e)}