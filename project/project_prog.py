import os
import cv2
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load and preprocess images from a folder
def process(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        # Load the image from the specified file
        img = cv2.imread(img_path)
        if img is not None:
            # Resize the image to a fixed size
            img = cv2.resize(img, (32, 32))
            # Convert the image to grayscale and flatten
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
            images.append(gray_img)
            labels.append(label)
    return images, labels

# A function to print The result in a table like format
def print_metrics(y_test, y_pred_knn, y_pred_dt, y_pred_log):
    print('\n' + ('-'*29) + f' {'Score Report'} ' + ('-'*29))
    print("{:<15} {:<20} {:<15} {:<15}".format('Metric', 'K-Nearest Neighbors', 'Decision Trees', 'Logistic Regression'))
    print("-" * 72)
    
    metrics = {
        'Accuracy': [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_log)],
        'Precision': [precision_score(y_test, y_pred_knn, average='weighted'), precision_score(y_test, y_pred_dt, average='weighted'), precision_score(y_test, y_pred_log, average='weighted')],
        'Recall': [recall_score(y_test, y_pred_knn, average='weighted'), recall_score(y_test, y_pred_dt, average='weighted'), recall_score(y_test, y_pred_log, average='weighted')],
        'F1-Score': [f1_score(y_test, y_pred_knn, average='weighted'), f1_score(y_test, y_pred_dt, average='weighted'), f1_score(y_test, y_pred_log, average='weighted')]
    }
    
    for metric, (knn_metric, dt_metric, log_metric) in metrics.items():
        print("{:<15} {:<20.4f} {:<15.4f} {:<15.4f}".format(metric, knn_metric, dt_metric, log_metric))

# Load and preprocess training data
yes_images, yes_labels = process('C:/Users/ahmed/Downloads/brain_mri/yes', 1)
no_images, no_labels = process('C:/Users/ahmed/Downloads/brain_mri/no', 0)

# Combine cat and dog data
X = np.array(yes_images + no_images)
y = np.array(yes_labels + no_labels)

# Randomly select a subset of features from the dataset
total_features = 1024
selected_features = random.randint(400, 1024)
selected_indices = random.sample(range(total_features), selected_features)
X = [[sublist[i] for i in selected_indices] for sublist in X]

# Normalize values
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Taking the last 15 records for future predictions
X_future = X[-15:]
X = X[:-15]
y_future = y[-15:]
y= y[:-15]

# Splitting the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

# Model 2: Decision Trees
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Model 3: Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# K-Fold Cross Validation  For each model
print()
models = [knn_model, dt_model, log_model]
model_names = ['K-Nearest Neighbors', 'Decision Tree', 'Logistic Regression']
for item in zip(models, model_names):
    scores = cross_val_score(item[0], X_train, y_train, cv=10, scoring='accuracy')
    print(item[1] + " Cross Validation Score: " + str(scores.mean())[:5])

# Getting the predicted values using the testing set
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
log_pred = log_model.predict(X_test)

# Printing out the metrics
print_metrics(y_test, knn_pred, dt_pred, log_pred)
print()