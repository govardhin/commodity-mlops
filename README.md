# Commodity Price Forecasting — End-to-End MLOps Pipeline

## 📌 Project Overview
This project demonstrates a complete MLOps pipeline for forecasting commodity price trends (Gold dataset used as a sample use case).

The system is designed to be dataset-agnostic and can work with any time-series commodity data (Crude Oil, Metals, etc.).

---

## 🚀 Architecture

Data Ingestion → Data Validation → Feature Engineering → Model Training → Evaluation → Model Versioning → FastAPI Serving → Docker Containerization → CI/CD (GitHub Actions)

---

## 🧠 Features

- Automated training pipeline
- Model evaluation & metrics tracking
- Model metadata logging
- FastAPI serving layer
- Dockerized deployment
- GitHub Actions CI pipeline
- Dataset-agnostic design

---

## 📊 Model Metrics (Sample)

- MAE
- RMSE
- R² Score

---

## 🐳 Run with Docker

```bash
docker-compose up --build