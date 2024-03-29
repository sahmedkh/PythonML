# Importing The necessary packages
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Providing data from the csv file and seperating features
df = pandas.read_csv("insurance.csv")
x = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Encoding categorical data to be numerical using one hot encoding
one_hot_x = pandas.get_dummies(x, columns = ['sex', 'smoker', 'region'])

# Splitting the dataset into training (70%) and testing (30%)
x_train, x_test, y_train, y_test = train_test_split(one_hot_x, y, test_size=0.3, random_state=0) 

# Creating and fitting the model
model = LinearRegression()
model.fit(x_train, y_train)

# Getting summary of the model
r_sq = model.score(x_train, y_train)
print("\nModel Summary: ")
print("Model score: ", r_sq.round(3))
print("Intercept: ", model.intercept_.round(3))
print("Coefficients: ", model.coef_.round(3))

# Predicting new values using the testing subset
y_pred = model.predict(x_test)
y_pred = [round(num, 2) for num in y_pred]
print("\nPredicted responses: ", y_pred)

# Score for the testing subset
r2_sq = model.score(x_test, y_test)
print("\nThe score for the testing subset: ", r2_sq.round(3))
print("\n")
