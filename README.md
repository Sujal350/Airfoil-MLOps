# Airfoil MLOps: Lift and Drag Prediction System

## Project Overview

This project presents an end-to-end MLOps pipeline for predicting the aerodynamic performance of NACA 4-digit airfoils using Machine Learning.

Traditionally, evaluating a new airfoil geometry requires:

* CAD Modeling
* Mesh Generation
* CFD Simulation
* Post-Processing

Each design iteration typically takes **45 minutes to 1 hour** depending on mesh quality, solver settings, and computational resources.

This application replaces the expensive CFD evaluation stage with trained machine learning models capable of predicting aerodynamic characteristics in milliseconds.

---

## Key Features

* NACA 4-Digit Airfoil Generation
* Automatic Airfoil Visualization
* Lift Prediction
* Drag Prediction
* CL/CD Ratio Calculation
* Performance Analysis from 0°–10° Angle of Attack
* Best Angle of Attack Identification
* PDF Report Generation
* DVC-Based Data Versioning
* MLflow Experiment Tracking (Local Development)
* Interactive Streamlit Web Application

---

## Time Savings Analysis

### Traditional CFD Workflow

| Step            | Typical Time |
| --------------- | ------------ |
| CAD Modeling    | 5–10 min     |
| Meshing         | 10–20 min    |
| CFD Simulation  | 20–30 min    |
| Post Processing | 5–10 min     |
| Total           | 45–60 min    |

### Machine Learning Prediction

Average Prediction Time:

**84.37 milliseconds**

### Performance Improvement

* Traditional CFD Time: ~3600 seconds
* ML Prediction Time: ~0.084 seconds
* Time Saved per Simulation: ~3599.92 seconds
* Speed Improvement: **99.9976% faster**

---

## Benefits

* Rapid aerodynamic design exploration
* Instant performance estimation
* Reduced computational cost
* Reduced energy consumption
* Faster design optimization
* Identification of optimum angle of attack
* Downloadable engineering reports

---

## Tech Stack

* Python
* Streamlit
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* DVC
* MLflow
* ReportLab

---

## Deployment

The application is deployed using Streamlit Community Cloud and provides real-time aerodynamic predictions through a web-based interface.

---

## Assumptions

All predictions are generated under the following project assumptions:

* Freestream Velocity = 50 m/s
* Wing Span = 1 m

Predictions should be interpreted within the operating range represented by the training dataset.

---

## Future Scope

* Multi-Airfoil Comparison
* XFOIL Integration
* Custom Airfoil Upload
* Docker Deployment
* Cloud-Based Experiment Tracking
* 3D Wing Performance Prediction
