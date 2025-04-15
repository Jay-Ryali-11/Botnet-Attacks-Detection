# Botnet Attacks Detection Web Application - README

## Overview

This project presents a comprehensive **Botnet Attacks Detection System** designed for **IoT environments** using a **hybrid machine learning model**. The system integrates multiple machine learning techniques, including **Artificial Neural Networks (ANN)**, **Recurrent Neural Networks (RNN)**, and **Long Short-Term Memory (LSTM)** networks, to detect and classify botnet activities effectively. A **web application** is developed to provide an intuitive interface for general users and authenticated users to interact with the detection models.

### Features:
- **User Registration and Login** for access control.
- **Dataset Upload** and management for model training.
- **Model Training** with ANN, RNN, and LSTM models, allowing users to select the best model based on performance.
- **Real-time Prediction** for classifying network traffic as botnet or normal activity.
- **LUCID Chatbot**: A conversational assistant integrated into the web application to guide users and answer queries.
- **Dynamic Model Evaluation**: After training, the system provides performance metrics and dynamically suggests the best model for the dataset.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Web Application Features](#web-application-features)
4. [Machine Learning Models](#machine-learning-models)
5. [Supported Input Fields](#supported-input-fields)
6. [Botnet Attack Categories](#botnet-attack-categories)
7. [Evaluation and Performance](#evaluation-and-performance)
8. [Future Improvements](#future-improvements)
9. [Contact](#contact)

---

## Installation

To set up and run the application locally, follow these steps:

### 1. **Clone the Repository**

To get the project up and running locally, first clone the repository:

```bash
git clone https://github.com/Jay-Ryali-11/Botnet-Attacks-Detection.git
cd Botnet-Attacks-Detection
```

### 2. **Create a Virtual Environment**

It is recommended to create a virtual environment to isolate project dependencies:

- For **Linux/macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- For **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. **Install Required Libraries**

Once the virtual environment is activated, install the necessary libraries by running the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually using:

```bash
pip install Flask==2.2.2 Flask-Mail==0.10.0 scikit-learn tensorflow mysql-connector-python pandas numpy re==2.2.1 werkzeug joblib torch nltk
```

### 4. **Set Up the Database**

If using **MySQL**, configure the database and user credentials. Alternatively, for local testing, you can use **SQLite**.

---

## Usage

### Running the Web Application

1. Start the Flask web application by running:

   ```bash
   python app.py
   ```

2. Open your browser and go to:

   `http://127.0.0.1:5000/` (default Flask address)

3. You can now interact with the web application.

---

## Web Application Features

### 1. **User Registration & Login**
   - **General users** can sign up and log in.
   - Users need to register with their email and password to gain access to advanced features.
   - **Support Pages** are available for assistance.

### 2. **Model Training**
   - Logged-in users can upload datasets to train models.
   - Users can choose from available models (**ANN**, **RNN**, **LSTM**) for training.
   - Model performance is assessed using metrics like accuracy, ROC-AUC, PR-AUC, and F1-score, allowing users to select the best model based on performance.

### 3. **Prediction Interface**
   - Users can input feature values to predict botnet activity.
   - The system provides real-time results classifying the network traffic as either **botnet** or **normal** and categorizes the type of attack.

### 4. **LUCID Chatbot**
   - **LUCID** is an intelligent conversational chatbot embedded within the web application.
   - LUCID helps users by:
     - Answering frequently asked questions related to botnet detection.
     - Guiding users through the application’s features.
     - Providing detailed information about the algorithms and models used in detection.
     - Assisting with dataset uploads, model training, and prediction steps.

### 5. **Dynamic Model Evaluation**
   - After training a model, the system dynamically evaluates its performance and provides feedback.
   - The system presents key performance metrics and suggests which model is best suited for the user’s dataset.

---

## Machine Learning Models

### **Artificial Neural Networks (ANN)**
   - A deep learning model that identifies complex non-linear relationships in network traffic data.
   - Ideal for general pattern recognition tasks.

### **Recurrent Neural Networks (RNN)**
   - Designed for sequential data, RNNs are effective in capturing patterns in time-series network traffic data.
   - They are capable of detecting botnet activities that depend on prior network events.

### **Long Short-Term Memory (LSTM) Networks**
   - An advanced type of RNN that excels at learning long-term dependencies in sequential data.
   - LSTMs are particularly suited for identifying time-dependent patterns in network traffic.

### **Stacking Approach**
   - The hybrid model uses a **stacking** approach, combining outputs from ANN, RNN, and LSTM models to improve detection accuracy.
   - This ensemble method aggregates the predictions from individual models, enhancing the system’s robustness and performance.

---

## Supported Input Fields

To make predictions, users need to provide the following network traffic input features via the web interface:

- **sbytes**: Source bytes (traffic sent by the source)
- **dbytes**: Destination bytes (traffic sent to the destination)
- **rate**: Data rate of the connection
- **sload**: Source load (requests per second)
- **dload**: Destination load (requests per second)
- **sinpkt**: Source packets per second
- **sjit**: Source jitter (variation in packet arrival times)
- **stcpb**: Source TCP flags
- **dtcpb**: Destination TCP flags
- **response_body_len**: Length of the response body (in bytes)

---

## Botnet Attack Categories

When a botnet attack is detected, the system classifies the activity into one of the following categories:

1. **Analysis**: Data mining or behavioral analysis traffic.
2. **Backdoor**: Malicious traffic indicating a backdoor attack.
3. **DoS**: Denial of Service (DoS) attack traffic, often flooding a system to overwhelm it.
4. **Exploits**: Malicious traffic that exploits system vulnerabilities.
5. **Fuzzers**: Traffic related to fuzz testing, often used to find vulnerabilities in systems.
6. **Generic**: General malicious behavior that doesn't fit other categories.
7. **Normal**: Legitimate network traffic without botnet indicators.
8. **Reconnaissance**: Traffic used for probing or mapping a network, often a precursor to an attack.
9. **Shellcode**: Malicious code executed as part of an exploit attack.
10. **Worms**: Self-replicating malicious software that spreads autonomously.

These categories help to identify and classify different types of botnet activity, improving the detection process and network security.

---

## Evaluation and Performance

- **Testing Accuracy**: 96.98%
- **ROC-AUC Score**: 0.9934
- **PR-AUC Score**: 0.9950
- **Additional Metrics**: Precision, Recall, F1-Score, and Training Time.

These performance metrics demonstrate that the hybrid machine learning model excels in detecting botnet activities in various IoT environments.

---

## Future Improvements

- **Additional Models**: Future versions will incorporate other machine learning models, such as **CNN** (Convolutional Neural Networks) and **XGBoost**, to improve detection accuracy further.
- **Real-Time Threat Monitoring**: Integrating the system with real-time monitoring tools for proactive botnet detection.
- **User Interface Enhancements**: Improvements to the user interface to provide a more dynamic and informative experience.
- **Scalability**: The system will be scaled to handle larger datasets and support more extensive IoT networks.

---

### Contact

For questions or suggestions, feel free to reach out to the project maintainers via [thecodexcipher@gmail.com].

---

This version of the README includes the updated **dynamic model evaluation**, more detailed explanations of the **machine learning models**, and improved formatting for clarity and structure. Let me know if you'd like to make further changes!
