# 1) Importing The necessary packages
import numpy
from sklearn.linear_model import LinearRegression

# 2) Providing data
x = numpy.array([23, 26, 30, 34, 43, 48, 52, 57, 58]).reshape((-1, 1))
y = numpy.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])

# 3) Creating and fitting the model
model = LinearRegression()
model.fit(x, y)

# 4) Getting results
r_sq = model.score(x, y)
print("Coeffecient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Slope: ", model.coef_)

# 5) Predict a response
y_pred = model.predict(numpy.array([40]).reshape((-1, 1)))
print("Predicted response: ", y_pred)