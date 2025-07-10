# Cardiovascular Disease Risk Prediction using Machine Learning

## Overview

Cardiovascular disease (CVD) remains a major global health challenge, emphasizing the importance of early detection and risk assessment. This project leverages state-of-the-art data science and machine learning techniques to predict the presence of CVD using a large-scale dataset of 70,000 patient records.

Our goal is to build an accurate, interpretable, and scalable predictive model to support early diagnosis and preventative care strategies.

## Features

* **Dataset**: 70,000 anonymized patient health records.
* **Feature Engineering**: Derived features such as Body Mass Index (BMI) and Mean Arterial Pressure (MAP) to enrich predictive power.
* **Preprocessing**: Included data standardization and dimensionality reduction using Principal Component Analysis (PCA).
* **Modeling Approaches**:

  * **Traditional Machine Learning**:

    * Logistic Regression
    * Support Vector Machines (SVM)
    * Gradient Boosting (XGBoost and LightGBM)
    * Gaussian Mixture Model-based Binary Classifier
  * **Ensemble Learning**:

    * Combined K-Nearest Neighbors (KNN), XGBoost, and LightGBM using a logistic regression meta-model
    * Achieved the highest test accuracy: **72.80%**
  * **Deep Learning**:

    * Feed Forward Neural Network (FFNN) with advanced optimizers (AdamW, OneCycleLR)
    * Accuracy: **72.80%**
  * **NLP Approach**:

    * Transformed tabular data into textual format and fine-tuned **BERT-base-uncased**
    * Accuracy: **70.93%**

## Key Results

* Achieved up to **72.80%** test accuracy using both ensemble methods and neural networks.
* Demonstrated the viability of combining traditional ML, deep learning, and NLP for healthcare prediction tasks.
* Provided an interpretable and scalable framework for early CVD detection.

## Future Work

* Integrate external data sources (e.g., lifestyle, genetics, medical history).
* Improve model performance and robustness with advanced neural architectures.
* Explore real-time deployment and clinical integration.

## Impact

This project contributes to the growing field of AI in healthcare by demonstrating how rigorous feature engineering, thoughtful model design, and hybrid modeling approaches can enhance disease risk prediction. It sets the foundation for developing reliable tools to assist clinicians and public health practitioners in identifying at-risk individuals earlier and more effectively.


