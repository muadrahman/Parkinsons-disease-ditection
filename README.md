# Parkinson's Disease Detection using Random Forest - Documentation

## Overview

This project aims to detect Parkinson's disease using vocal features with a Random Forest classifier. The dataset contains various vocal attributes that are analyzed to classify individuals as healthy or having Parkinson's disease.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Library Dependencies](#library-dependencies)
3. [Steps](#steps)
4. [Model Building](#model-building)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Feature Importance](#feature-importance)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Technologies Used](#technologies-used)
10. [Contact](#contact)
11. [References](#references)
12. [Project Link](#project-link)

## Dataset Description

- **Source:** The dataset is from the UCI Machine Learning Repository.
- **Structure:** The dataset contains 195 rows and 24 columns.
  - **Features:** Various vocal features (e.g., `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`)
  - **Target Variable:** `status` (1 indicates Parkinson's disease, 0 indicates healthy)

## Library Dependencies

The project requires several Python libraries, including pandas, numpy, scikit-learn, seaborn, and matplotlib. These libraries are used for data manipulation, machine learning model building, and visualization.

## Steps

### 1. Load and Prepare Data
1. Load the dataset from a CSV file.
2. Remove the `name` column as it is not needed for the analysis.
3. Define features (X) and target variable (Y).
4. Split the data into training and testing sets.

### 2. Build Initial Model
1. Train a Random Forest classifier on the training data.
2. Evaluate the initial model's performance on the test data.

### 3. Hyperparameter Tuning
1. Use Grid Search to find the best hyperparameters for the Random Forest model.
2. Train the model with the optimal hyperparameters.
3. Evaluate the tuned model's performance.

### 4. Feature Importance and Selection
1. Determine the importance of each feature using the trained Random Forest model.
2. Select the most important features.
3. Retrain the Random Forest model using only the selected features.
4. Evaluate the model with selected features.

### 5. Final Evaluation and Results
1. Compare the performance of the initial, tuned, and feature-selected models.
2. Analyze and report the results.

## Model Building

A Random Forest classifier is trained on the training data. This model is chosen for its robustness and ability to handle high-dimensional data.

## Hyperparameter Tuning

Grid Search is used to find the best hyperparameters for the Random Forest model. This involves testing various combinations of parameters to find the optimal settings that improve model performance.

## Feature Importance

The importance of each feature is determined using the trained Random Forest model. The most important features are selected for further analysis to simplify the model and potentially improve its performance.

## Evaluation

The model's performance is evaluated on the test data using several metrics, including accuracy, confusion matrix, and classification report. Both the tuned model and the model with selected features are evaluated to compare their performance.

## Results

- **Initial Model Accuracy:** The accuracy of the initial model before tuning is 84.6%.
- **Accuracy After Tuning:** Hyperparameter tuning improves the accuracy to 87.2%.
- **Accuracy After Feature Selection:** The accuracy improves to 92.3% after selecting the most important features.
- **F1 Score for Healthy Individuals:** 80%
- **F1 Score for Diagnosing Parkinson's Disease:** 95.2%

Hyperparameter tuning and feature selection lead to a significant improvement in the model's performance, demonstrating the importance of these steps in the machine learning process.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib

## Contact

For inquiries or feedback, feel free to reach out:
- [Gmail](mailto:mr.muadrahman@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/muadrahman/)

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

## Project Link

For further details and access to the project repository, visit [this link](https://github.com/muadrahman/Parkinsons-disease-ditection).
