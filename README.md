# Iris Flower Classification Using AI 🌸

## Project 2 — DecodeLabs AI Internship

This project is a supervised machine learning classification system that predicts the species of an Iris flower based on its physical measurements.

The model is trained using the Iris dataset and uses the **K-Nearest Neighbors (KNN)** classification algorithm.

---

## 🎯 Objective

The objective of this project is to understand the basic workflow of supervised machine learning by:

- Loading and understanding a dataset
- Separating features and target values
- Splitting data into training and testing sets
- Scaling the features
- Training a classification model
- Making predictions
- Evaluating the model
- Predicting the species of a new flower

---

## 📊 Dataset

The project uses the **Iris dataset**, which contains:

- 150 samples
- 4 features
- 3 classes

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Classes

- Setosa
- Versicolor
- Virginica

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN classifies a new data point by looking at the nearest data points from the training dataset.

In this project:

⚙️ Machine Learning Workflow
Iris Dataset
     ↓
Data Understanding
     ↓
Feature & Target Separation
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
KNN Model Training
     ↓
Prediction
     ↓
Model Evaluation
     ↓
New Flower Classification
📈 Train/Test Split

The dataset is divided into:

80% → Training Data
20% → Testing Data

The training data is used to train the model, while the testing data is used to evaluate how well the trained model performs on unseen data.

📏 Feature Scaling

StandardScaler is used to standardize the feature values before applying KNN.

The scaler is fitted using the training data and then applied to both the training and testing data.

📊 Model Evaluation

The model is evaluated using:

Accuracy
Confusion Matrix
F1 Score
Results
Accuracy: 1.0

F1 Score: 1.0
Confusion Matrix
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

The model correctly classified all 30 samples in the test set.

🔍 New Flower Prediction

The program also allows the user to enter measurements for a new flower:

Sepal Length
Sepal Width
Petal Length
Petal Width

The trained KNN model then predicts the species of the flower.

Example:

Enter measurements for a new flower:

Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2

Predicted species: setosa
🛠️ Technologies Used
Python
Pandas
Scikit-learn
Matplotlib
Seaborn
Number of Neighbors (K) = 5
```text
