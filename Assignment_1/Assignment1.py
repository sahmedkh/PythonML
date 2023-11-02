
# Importing The necessary packages
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Providing data from the csv file and dividing it
df = pandas.read_csv("insurance.csv")

x = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
one_hot_x = pandas.get_dummies(x, columns = ['sex', 'smoker', 'region'])
y = df['charges']
x_train, x_test, y_train, y_test = train_test_split(one_hot_x, y, test_size=0.3, random_state=0) 

# Creating and fitting the model
model = LinearRegression()
model.fit(x_train, y_train)

# Getting summary of the model
r_sq = model.score(x_train, y_train)
print("Model score: ", r_sq)
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)

# Predicting new values using the testing subset
y_pred = model.predict(x_test)
y_pred = [round(num, 2) for num in y_pred]
print("Predicted responses: ", y_pred)

# Score for the testing subset
r2_sq = model.score(x_test, y_test)
print("The score for the testing subset: ", r2_sq)
