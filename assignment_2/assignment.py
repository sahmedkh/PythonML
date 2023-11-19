# Importing The necessary packages
import pandas
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# Providing data from the csv file and seperating features
df = pandas.read_csv("C:/Users/ahmed/OneDrive/Documents/Code/PythonML/assignment_2/iris.csv", header=None)
x = df.iloc[:, :4]
y = df.iloc[:, 4]

# Splitting the dataset into training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) 

# Fitting a logistic regression model
logr = linear_model.LogisticRegression()
logr.fit(x_train,y_train)

# Getting the classification report on the test data
print()
print("Summary of the logistic regression model")
print(classification_report(y_test, logr.predict(x_test)))

# Fitting a decision tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Getting the classification report on the test data
print()
print("Summary of the decision tree model")
print(classification_report(y_test, clf.predict(x_test)))

# Fitting a KNN model
nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs = nbrs.fit(x_train, y_train)

# Getting the classification report on the test data
print()
print("Summary of the K nearest neighbors model")
print(classification_report(y_test, nbrs.predict(x_test)))