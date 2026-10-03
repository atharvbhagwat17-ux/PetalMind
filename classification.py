
# Project 2 - Data Classification Using AI
# Iris Flower Classification using KNN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


iris= load_iris()

X=iris.data 
y=iris.target

df=pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y
df["species"] = [iris.target_names[i] for i in y]

print("First 5 rows:")
print(df.head())

print("\nNumber of samples:", len(X))
print("Number of features", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.2, 
    random_state=42
    )

print("\nTraining samples:", len(X_train))
print("Testing samples", len(X_test))

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

f1 = f1_score(y_test, y_pred, average='weighted')
print("\nF1 Score:", f1)

plt.figure(figsize=(6,5))

sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names
        )

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN Classifier')

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Dataset")

plt.show()

print("\nEnter measurements for a new flower:")

sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

predicted_species = iris.target_names[prediction[0]]

print("Predicted species:", predicted_species)