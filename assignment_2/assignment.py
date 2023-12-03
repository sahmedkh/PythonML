# Importing The necessary packages
import pandas
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# Function to print metrics in a table format
def print_metrics(name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    print('\n' + ('-'*15) + f' {name} ' + ('-'*14))
    print("{:<20} {:<15} {:<15}".format('Metric', 'Training', 'Testing'))
    print("-" * 50)
    
    metrics = {
        'Accuracy': [accuracy_score(y_true_train, y_pred_train), accuracy_score(y_true_test, y_pred_test)],
        'Precision': [precision_score(y_true_train, y_pred_train, average='weighted'), precision_score(y_true_test, y_pred_test, average='weighted')],
        'Recall': [recall_score(y_true_train, y_pred_train, average='weighted'), recall_score(y_true_test, y_pred_test, average='weighted')],
        'F1-Score': [f1_score(y_true_train, y_pred_train, average='weighted'), f1_score(y_true_test, y_pred_test, average='weighted')]
    }
    
    for metric, (train_metric, test_metric) in metrics.items():
        print("{:<20} {:<15.4f} {:<15.4f}".format(metric, train_metric, test_metric))

# Providing data from the csv file and seperating features
df = pandas.read_csv("C:/Users/ahmed/OneDrive/Documents/Code/PythonML/assignment_2/iris.csv", header=None)
x = df.iloc[:, :4]
y = df.iloc[:, 4]

# Splitting the dataset into training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99) 

# Fitting a logistic regression model
logr = linear_model.LogisticRegression()
logr.fit(x_train,y_train)
logr_y_pred_train = logr.predict(x_train)
logr_y_pred_test = logr.predict(x_test)

# Printing metrics for logistic regression
print_metrics('Logistic Regression', y_train, logr_y_pred_train, y_test, logr_y_pred_test)

# Fitting a decision tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
clf_y_pred_train = clf.predict(x_train)
clf_y_pred_test = clf.predict(x_test)

# Printing metrics for decision trees
print_metrics('Decision Trees', y_train, clf_y_pred_train, y_test, clf_y_pred_test)

# Fitting a KNN model with k = 5
nbrs = KNeighborsClassifier()
nbrs = nbrs.fit(x_train, y_train)
nbrs_y_pred_train = nbrs.predict(x_train)
nbrs_y_pred_test = nbrs.predict(x_test)

# Printing metrics for K-Nearest Neighbors
print_metrics('K-Nearest Neighbors', y_train, nbrs_y_pred_train, y_test, nbrs_y_pred_test)