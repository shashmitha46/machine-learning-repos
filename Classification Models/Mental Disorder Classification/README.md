# Mental Disorder Classification

This project focuses on using machine learning techniques to classify different types of mental disorders based on a dataset of psychological and behavioral attributes.

## Project Overview

Mental health issues are rising globally, and it's crucial to have tools that can help professionals classify mental disorders accurately. This project applies various classification algorithms to analyze mental health data and make predictions about the type of mental disorder a person might be experiencing. The models can be used to assist healthcare providers in making informed decisions based on patient data.

The main goals of this project are:
- To build accurate machine learning models for classifying mental disorders.
- To evaluate the performance of different algorithms and choose the best one for deployment.
- To explore how different features influence the classification results.

## Dataset

The dataset contains various features related to mental health, including demographic details and psychological factors such as:
- Age, gender
- History of mental health issues
- Psychological assessment scores
- Behavioral patterns (e.g., sleep, diet, activity level)

*Note: The actual dataset is not provided in this repository due to privacy concerns. You can use synthetic data or publicly available datasets related to mental health for experimentation.*

## Features

- **Data Preprocessing:** Cleaning the data, handling missing values, and normalizing features for better model performance.
- **Feature Selection:** Selecting the most relevant features that impact mental health classifications.
- **Model Building:** Implementing a variety of machine learning classification algorithms to predict mental disorder categories.
- **Model Evaluation:** Using different metrics like accuracy, precision, recall, and F1 score to evaluate model performance.

## Models Implemented

Several machine learning models have been implemented and compared in this project:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

Each model is trained and tested on the dataset, with a comparison of their respective performances.

## Evaluation Metrics

The models are evaluated based on:
- **Accuracy:** How often the classifier correctly identifies a mental disorder.
- **Precision:** How many of the predicted positive cases were actually positive.
- **Recall:** How well the classifier identifies true positive cases.
- **F1 Score:** The balance between precision and recall.

## Results

The project shows that the **Random Forest Classifier** performed the best, achieving an accuracy of around 88%. Other models like Logistic Regression and Support Vector Machines also showed promising results but with slightly lower accuracy.

A summary of the results:

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 85%      | 0.84      | 0.83   | 0.83     |
| Decision Tree         | 80%      | 0.79      | 0.78   | 0.78     |
| Random Forest         | 88%      | 0.87      | 0.86   | 0.86     |
| SVM                   | 84%      | 0.83      | 0.82   | 0.82     |
| KNN                   | 82%      | 0.81      | 0.80   | 0.80     |

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/shashmitha46/machine-learning-repos.git
    ```

2. Navigate to the `Mental Disorder Classification` folder:
    ```bash
    cd machine-learning-repos/Classification\ Models/Mental\ Disorder\ Classification
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook or Python scripts to train and test the models:
    - Jupyter Notebook:
        ```bash
        jupyter notebook notebooks/Mental_Disorder_Classification.ipynb
        ```
    - Python Script:
        ```bash
        python scripts/classify_mental_disorder.py
        ```

