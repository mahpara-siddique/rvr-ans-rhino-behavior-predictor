# 🦏 **RhinoVisitorResponse Artificial Neural System (RVR-ANS)**

### *A Deep-Learning Powered ANN Model for Predicting White Rhinoceros Visitor Interaction Behavior*

---

##  **Overview**

**RVR-ANS (RhinoVisitorResponse Artificial Neural System)** is a **deep-learning based predictive system** developed to forecast visitor–rhino interaction behavior using detailed ethological data of a captive **White Rhinoceros** housed at **Lahore Zoo**.

Built using **TensorFlow’s Artificial Neural Network (ANN)** architecture, the system analyzes multiple behavioral metrics—such as foraging patterns, roaming activity, resting durations, and feeding events—to predict how the rhino responds to human presence.

The goal is to support:

* Zoo welfare management
* Behavioral analysis
* Visitor engagement optimization
* Scientific data-driven decision-making

---
![ss1](https://github.com/user-attachments/assets/9b108d84-ed4f-463a-bbdf-1ad8be38b907)


![ss2](https://github.com/user-attachments/assets/397843e8-74df-405d-bf5a-a708eb6c9444)


![ss3](https://github.com/user-attachments/assets/27cdd4ff-6e46-4ab5-9909-25207a4f203b)

##  **Research Objective**
To determine:

> **If a rhinoceros remains under behavioral conditions similar to those observed from May 11 to August 11, how will it interact with visitors?**

The ANN model predicts categories of behavioral responses such as:

* **Visitor-Directed Approach (VDA)**
* **Moderate Interaction (MI)**
* **Lack of Response (LKR)**
* **Avoidant / Low Keeper Response (LKI)**

This project merges **ethology**, **deep learning**, and **zoo animal welfare science**.

---

##  **Dataset Description**

The dataset consists of **3 months of continuous behavioral observation** of the same White Rhinoceros at Lahore Zoo. Behavior was recorded using systematic sampling techniques.

### **Input Features (X):**

* Total Resting Duration (minutes)
* Morning Rest Frequency
* Foraging Frequency
* Morning Foraging Count
* Roaming Frequency (Outdoor enclosure)
* Evening Roaming Frequency (Indoor enclosure)
* Provisioned Feeding Events
* Standing Still Duration (minutes)

Target Variable (Y):

Visitor Interaction Response Category

Visitor-Directed Approach (VDA)

Moderate Interaction (MI)

Lack of Response (LKR)

Avoidant / Low-Keeper Interaction (LKI)

These metrics were carefully normalized and engineered for ANN training.

---

##  **Deep Learning Model: RVR-ANS Architecture**

RVR-ANS is implemented as a **multi-layer Artificial Neural Network (ANN)** using **TensorFlow/Keras**.

### **Model Components & Their Purpose**

| Component             | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| **Dense Layers**      | Learn non-linear behavioral relationships         |
| **Dropout Layers**    | Prevent overfitting & improve generalization      |
| **L1 Regularization** | Adds sparsity & stabilizes model weights          |
| **ReLU Activation**   | Efficient training on non-linear ethological data |
| **Adam Optimizer**    | Fast convergence on deep-learning tasks           |
| **EarlyStopping**     | Avoids overfitting during long training cycles    |
| **Epochs = 200**      | Ensures deep feature extraction                   |

The trained model is stored as a **.pkl object** using Python’s pickle for deployment.

---

##  **Technologies & Libraries**

```python
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
```

Additional tools:

* TensorFlow/Keras (Deep Learning)
* Scikit-learn (Scaling & splitting)
* Flask (Deployment)

---

## **Model Training Pipeline**

### **1. Data Preprocessing**

* Cleaned and validated raw behavioral logs
* Encoded categorical interaction labels
* Normalized all quantitative inputs using StandardScaler

### **2. Feature Engineering**

* Split into input features (X) and target labels (Y)
* Ensured ANN-friendly 2D tensor format

### **3. Scaling & Train/Test Split**

ANNs perform best with normalized inputs → scaling applied
Dataset split for objective performance validation

### **4. Building the ANN**

* Defined hidden layers
* Applied dropout
* Added L1 regularization
* Used ReLU activations

### **5. Training**

* 200 epochs with monitoring
* EarlyStopping triggers when validation loss plateaus
* ANN learns complex, non-linear ethogram patterns

### **6. Evaluation**

Model performance measured via:

* Validation loss
* Accuracy
* Mean Absolute Error (MAE)

### **7. Future Prediction**

Custom interface allows entering new behavioral values to simulate rhino responses.

---

## 🌐 **Deployment with Flask**

A simple and intuitive web interface allows users to:

* Input behavioral metrics
* Run real-time ANN predictions
* View interpreted interaction categories

The frontend includes:

* Input grid
* Behavior metric guidelines
* Prediction output section

---

## 🚀 **How to Run Locally**

### **1. Clone repository**

```bash
git clone https://github.com/mahpara-siddique/rvr-ans-rhino-behavior-predictor.git
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Flask app**

```bash
python app.py
```

### **4. Open browser**

```
http://127.0.0.1:5000/
```

## **Acknowledgements**

* Lahore Zoo authorities
* White Rhinoceros (study individual)
* Supervisors and wildlife experts
* Open-source deep learning community


  
