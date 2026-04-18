# Botnet Attacks Detection System v2.0 - Production Architecture

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.2.2-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-NLP-red)
![MySQL](https://img.shields.io/badge/MySQL-8.0-blue)

## Overview

This project presents a comprehensive, production-grade **Botnet Attacks Detection System** designed for **IoT environments**. Version 2.0 introduces a major architectural refactor, transitioning from a monolithic application to a highly scalable, secure, and modular **Flask Application Factory** pattern. 

The system integrates hybrid machine learning techniques—including **Artificial Neural Networks (ANN)**, **Recurrent Neural Networks (RNN)**, and **Long Short-Term Memory (LSTM)** networks—to detect and classify botnet activities in network traffic with high accuracy (~96.98%).

### 🚀 What's New in V2.0 (Enterprise Refactor)

* **Modular Architecture (Blueprints):** Routing logic is now decoupled into specific domains (`auth`, `ml`, `chatbot`, `general`) for maintainability.
* **Security by Design:** Implementation of `.env` for secrets management, `bcrypt` for password hashing, and CSRF protections.
* **Decoupled ML Engine:** Machine learning prediction (`predictor.py`), validation (`validator.py`), and training (`pipeline.py`) are strictly separated from web routing.
* **Database Connection Pooling:** Implemented thread-safe MySQL connection pooling (`utils/db.py`) to prevent race conditions and timeouts under load.
* **Test-Driven Foundation:** Introduced a `tests/` directory for unit testing authentication and prediction pipelines.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation & Setup](#installation--setup)
3. [Usage](#usage)
4. [Web Application Features](#web-application-features)
5. [Machine Learning Models](#machine-learning-models)
6. [Botnet Attack Categories](#botnet-attack-categories)
7. [Evaluation and Performance](#evaluation-and-performance)
8. [Future Improvements](#future-improvements)

---

## Project Structure

The application follows a strict separation of concerns, ensuring scalability and ease of testing:

    Botnet-Attacks-Detection/
    ├── app/                        # Core Application Package
    │   ├── routes/                 # Flask Blueprints (auth, ml, chatbot, general)
    │   ├── models/                 # Database schema models
    │   ├── ml/                     # ML Engine (pipeline, predictor, validator)
    │   ├── chatbot/                # LUCID PyTorch NLP Engine
    │   ├── utils/                  # DB connection pools, auth helpers, loggers
    │   └── templates/              # Jinja2 HTML Templates
    ├── saved_models/               # Pre-trained .h5 model weights
    ├── data/                       # Datasets (e.g., UNSW_NB15.csv)
    ├── chatbot_data/               # NLP intents and .pth weights
    ├── static/                     # CSS, JS, and Images
    ├── tests/                      # Unit and integration tests
    ├── run.py                      # Application entry point
    ├── config.py                   # Environment-specific configurations
    ├── .env.example                # Template for environment variables
    └── requirements.txt            # Python dependencies

---

## Installation & Setup

To set up and run the V2.0 application locally, follow these steps:

### 1. Clone the Repository

    git clone https://github.com/Jay-Ryali-11/Botnet-Attacks-Detection.git
    cd Botnet-Attacks-Detection

### 2. Create a Virtual Environment

Isolate your project dependencies:

**Linux/macOS**
    python3 -m venv venv
    source venv/bin/activate

**Windows**
    python -m venv venv
    venv\Scripts\activate

### 3. Install Required Libraries

    pip install -r requirements.txt

### 4. Environment Variables Configuration (Crucial)

V2.0 strictly enforces secrets management. You must create a `.env` file:

1. Copy the example file:
    cp .env.example .env

2. Open `.env` and fill in your specific database credentials, Flask secret keys, and mail server details.

### 5. Database Setup

Initialize your MySQL database using the provided schema. *Note: The SQL file no longer contains hardcoded credentials.*

    mysql -u your_user -p < db_setup.sql

---

## Usage

### Running the Web Application

Unlike V1, the application is now launched via the entry point file `run.py`.

    python run.py

Access the application in your browser at: `http://127.0.0.1:5000/`

### Running the Tests

To ensure the integrity of the authentication and ML pipelines, execute the test suite:

    pytest tests/

---

## Web Application Features

### 1. **User Registration & Login**
* **Access Control:** General users can sign up and log in. Secure features require email/password authentication.
* **Support Pages:** Dedicated UI views for user assistance and recovery.

### 2. **Model Training**
* **Dataset Uploads:** Logged-in users can upload datasets to train models dynamically.
* **Custom Training:** Users can choose from available models (**ANN**, **RNN**, **LSTM**) for training.
* **Performance Metrics:** The system outputs accuracy, ROC-AUC, PR-AUC, and F1-score, allowing users to select the best model.

### 3. **Prediction Interface**
* **Real-Time Inference:** Users can input feature values to predict botnet activity.
* **Classification:** The system provides real-time results classifying the network traffic as either **botnet** or **normal** and categorizes the specific type of attack.

### 4. **LUCID Chatbot**
* **Integrated Assistant:** An intelligent conversational chatbot embedded within the web application.
* **Capabilities:** Answers FAQs related to botnet detection, guides users through application features, provides algorithm details, and assists with the dataset upload and training steps.

### 5. **Dynamic Model Evaluation**
* After training a model, the system dynamically evaluates its performance, presents key metrics, and provides automated feedback on which model is best suited for the user’s specific dataset.

---

## Machine Learning Models

The hybrid model uses a **stacking** approach, combining outputs to improve detection accuracy and handle complex time-series network data.

* **Artificial Neural Networks (ANN):** Best for rapid inference and identifying complex non-linear relationships in static network traffic snapshots.
* **Recurrent Neural Networks (RNN):** Designed for sequential data, effective in capturing patterns in time-series network traffic data.
* **Long Short-Term Memory (LSTM) Networks:** Our highest-performing model. LSTMs utilize a gating mechanism to learn long-term dependencies, making them exceptionally suited for identifying slow, sequential botnet attacks.

### Supported Input Features (UNSW-NB15 Benchmark)

The models analyze the following flow-level metrics:

`sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sinpkt`, `sjit`, `stcpb`, `dtcpb`, `response_body_len`

---

## Botnet Attack Categories

When an anomaly is detected, the inference engine classifies the flow into one of the following 10 categories:

1. **Analysis**
2. **Backdoor**
3. **DoS (Denial of Service)**
4. **Exploits**
5. **Fuzzers**
6. **Generic**
7. **Normal**
8. **Reconnaissance**
9. **Shellcode**
10. **Worms**

---

## Evaluation and Performance

* **Testing Accuracy (LSTM):** ~96.98%
* **ROC-AUC Score:** 0.9934
* **PR-AUC Score:** 0.9950
* **Inference Latency:** Minimized via asynchronous model pre-loading on user login, allowing for near-instantaneous predictions.

---

## Future Improvements (V3 Roadmap)

* **Containerization & Orchestration:** Full integration with Docker and `docker-compose` to containerize the Flask application, ML engine, and MySQL database for isolated, reproducible deployments, paving the way for Kubernetes orchestration.
* **Real-Time Automated Inference:** Transition from manual HTTP submissions to an event-driven streaming pipeline (e.g., Apache Kafka + Zeek) for 24/7 autonomous packet monitoring.
* **Adaptive MLOps Lifecycle:** Implement CI/CD pipelines for automated model retraining to combat concept drift and zero-day variants.
* **System Observability:** Integrate Prometheus and Grafana for centralized monitoring of model drift and system health.

---

## Contact

For questions, discussions, or professional inquiries, feel free to reach out:

* **Email:** [thecodexcipher@gmail.com](mailto:thecodexcipher@gmail.com)
* **LinkedIn:** [Jaya Sai Sri Vardhan Ryali](https://linkedin.com/in/jaya-sai-sri-vardhan-ryali-25bb76237)