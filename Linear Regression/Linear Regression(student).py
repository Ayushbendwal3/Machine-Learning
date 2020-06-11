import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import random

data = pd.read_csv("Linear Regression/Datasets/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

linear = LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy:", acc*100, "%")

predictions = linear.predict(x_test)

ran = random.randint(1, len(predictions))

print("Predicted value: "+str(predictions[ran]).upper())

print("Actual Value: "+str(y_test[ran]).upper())
