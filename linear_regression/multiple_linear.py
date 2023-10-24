# 1) Importing The necessary packages
import numpy
from sklearn.linear_model import LinearRegression

# 2) Providing data
x = [ 
    [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
    ]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = numpy.array(x), numpy.array(y)

# 3) Creating and fitting the model
model = LinearRegression()
model.fit(x, y)

# 4) Getting results
r_sq = model.score(x, y)
print("Coeffecient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)

# 5) Predict a response
y_pred = model.predict(numpy.array([40, 9]).reshape((-1, 2)))
print("Predicted response: ", y_pred)