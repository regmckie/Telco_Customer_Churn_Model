# Telco Customer Churn Model Project
---

FILENAME: README.txt <br>
AUTHOR: Reg Gonzalez <br>
CONTACT: regmckie@gmail.com <br>
DATE: 4/21/2026

---

### PROJECT DESCRIPTION:

This project built an end-to-end solution for predicting customer churn for the telecommunications industry. This process went from data prep/exploration and modeling to using FastAPI, containerization in Docker, and deploying the model via AWS.

---

### CODE AND RESOURCES:

**Python Version:** 3.11 <br>
**Packages:** MLFlow, Pydantic, Pandas, FastAPI, Gradio, Matplotlib, etc. (see more in `requirements.txt`) <br>
**Installing Requirements:** `pip install -r requirements.txt` <br>
**Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### PRACTICAL APPLICATIONS OF PROJECT:
- Companies can make faster decisions: Having a system that predicts customer churn is important because it allows companies to act and make necessary adjustments before their clients leave
- Operational ML model: The model can be accessed via the deployment of the FastAPI + web UI on AWS; you don't have to use the Jupyter notebook
- Repeatable results: Use of GitHub Actions (CI/CD) and containers in Docker makes it so that anyone can rebuild and test the model in whatever environment they have
- Traceable experiments: MLflow allows the tracking of multiple models & their metrics; you can easily reproduce models or go back to earlier models if needed

---

### PROJECT ASPECTS:

- **Data Exploration and Modeling:** EDA; feature engineering; RandomForest, LightGBM, and XGBoost classifiers
- **ML Model Tracking:** Model runs, metrics, and artifacts logged under an MLflow experiment
- **Inference Service:** Ensures serving-time feature transformations exactly match training-time transformations; critical for model accuracy and reproduction
- **Web UI:** Gradio interface for user-friendly testing of model
- **Containerization:** Docker image with uvicorn entrypoint listening on port 8000; necessary to make the model and the enviornment + dependencies work on anyone else's system
- **CI/CD:** GitHub Actions builds the image and pushes it to Docker Hub
- **Orchestration:** AWS ECS Fargate runs the container
- **Network:** Application Load Balancer (ALB) on HTTP:80 forwards to a Target Group (IP targets on HTTP:8000)
- **Security:** Sets up a secure, tiered network architecture where the ALB receives public internet traffic, while the backend tasks (ECS/EC2) are locked down to only accept traffic from the ALB; enforces that no traffic can bypass the ALB and access the backend directly
- **Observability:** CloudWatch logs for stdout/stderr and ECS service events

---

### DEPLOYMENT FLOW:

1. Push to main --> GitHub Actions builds the Docker image and pushes it to Docker Hub
2. ECS service is updated (either manually or via the workflow) to force a new deployment
3. ALB health checks hit on container's root path (/) on port 8000; once healthy, traffic is routed to the new task
4. Users call POST (predict) or open the Gradio UI via the ALB DNS to use the model to make customer churn predictions
