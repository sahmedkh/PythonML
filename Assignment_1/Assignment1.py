# Importing The necessary packages
import numpy
import csv
from sklearn.linear_model import LinearRegression

# Variable declarations
x = []
y = []
positions = {}
i = 1

# Providing data from the csv file
with open('C:/Users/ahmed/OneDrive/Documents/Code/PythonML/linear_regression/salary.csv', mode ='r') as file:
    next(file)
    csvFile = csv.reader(file)
    for lines in csvFile:
        if lines[0] not in positions.values():   
            positions[i] = lines[0]
            i+=1 
        x.append([i, int(lines[1])])
        y.append(int(lines[2]))
x, y = numpy.array(x), numpy.array(y)

# Creating and fitting the model
model = LinearRegression()
model.fit(x[0:7], y[0:7])

# Getting results
r_sq = model.score(x[0:7], y[0:7])
print("Coeffecient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)

# Predicting a response
y_pred = model.predict(numpy.array(x[7:10]).reshape((-1, 2)))
print("Predicted response: ", y_pred)
