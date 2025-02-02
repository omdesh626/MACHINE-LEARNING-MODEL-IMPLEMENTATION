# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY:CODTECH IT SOLUTIONS

NAME:om suresh deshmukh

INTERN ID:CT08KWT

DOMAIN:PYTHON PROGRAMMING

DURATION:4 WEEKS

MENTOR:NELLA SANTOSH

# Titanic Survival Prediction with Logistic Regression

## Overview
This project demonstrates how to use Logistic Regression to predict survival outcomes for passengers aboard the Titanic using the Titanic dataset from sklearn.

## Dataset Description
You are provided with:
1. **Training Dataset (`training_titanic_x_y_train.csv`)**: Contains features (`X_train`) and labels (`Y_train`).
2. **Test Dataset (`test_titanic_x_test.csv`)**: Contains features (`X_test`) for which predictions are required.

## Objective
Your task is to:
1. Train a Logistic Regression model using the provided training data.
2. Generate predictions for the given test data.
3. Save predictions in a CSV file without headers, with only one column containing the prediction results.

## Instructions
1. **Algorithm**: Use Logistic Regression as the training algorithm.
2. **Data Format**: Files are in CSV format.
3. **Submission Format**: Output a CSV file containing only the predictions for the test data, without headers.
4. **Evaluation**: The score is based on the number of accurate predictions.

## Code Explanation
### Step 1: Load the Data
```python
import numpy as np
x_test = np.genfromtxt('test_titanic_x_test.csv', delimiter=',', skip_header=1)
train = np.genfromtxt('training_titanic_x_y_train.csv', delimiter=',', skip_header=1)

x_train = train[:, :-1]
y_train = train[:, -1]
```
- The training data is split into `x_train` (features) and `y_train` (labels).
- Test data (`x_test`) is loaded for making predictions.

### Step 2: Handle Missing Data
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)
```
- Missing values in the dataset are imputed using the mean strategy.

### Step 3: Train the Model
```python
from sklearn.linear_model import LogisticRegression
alg1 = LogisticRegression(C=2.0)
alg1.fit(x_train, y_train)
```
- A Logistic Regression model with regularization parameter `C=2.0` is trained on the data.

### Step 4: Make Predictions
```python
predictions = alg1.predict(x_test)
```
- The trained model generates predictions for the test dataset.

### Step 5: Save Predictions
```python
np.savetxt('prediction.csv', predictions, fmt='%d', delimiter=',')
```
- The predictions are saved in a CSV file named `prediction.csv` without headers.

## Usage Instructions
1. Ensure that `training_titanic_x_y_train.csv` and `test_titanic_x_test.csv` are in the working directory.
2. Run the script to train the model and generate predictions.
3. Submit the generated `prediction.csv` file for evaluation.

## Requirements
- Python 3.x
- NumPy
- scikit-learn

### Installation
Install the required packages using pip:
```bash
pip install numpy scikit-learn
```
OUTPUT-
![Image](https://github.com/user-attachments/assets/412b9a0e-738c-4b46-8cdb-923f0b56dc5b)

