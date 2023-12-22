# Import the necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

# Providing data from the csv file and seperating features
df = pd.read_csv("diabetes.csv")
x = df.iloc[:, :8]
y = df.iloc[:, 8]

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fitting the Naive-Bayes model
nb = GaussianNB()
nb.fit(x_train, y_train)

# Fitting the sequential ANN (default is 3 layers: input, 1 hidden with 100 neurons, and output)
ann = MLPClassifier(random_state=42)
ann.fit(x_train, y_train)

# Testing Naive-Bayes model
nb_pred = nb.predict(x_test)
print("Naive-Bayes Model Accuracy = ", accuracy_score(y_test, nb_pred))

# Testing the sequential ANN
ann_pred = ann.predict(x_test)
print("Artificial Neural Network Accuracy = ", accuracy_score(y_test, ann_pred))