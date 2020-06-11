import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import random
from matplotlib import pyplot as plt


data = pd.read_csv("Linear Regression/Datasets/forestfires.csv")

data = data[["X", "Y", "month", "FFMC", "DMC", "DC",
             "ISI", "temp", "RH", "wind", "rain", "area"]]

months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

data.month = data.month.map(months)

predict = "month"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy:", acc*100, "%")

predictions = linear.predict(x_test)

key_list = list(months.keys())
val_list = list(months.values())

ran = random.randint(1, len(predictions))

print("Predicted value: "+str(key_list[val_list.index(
    round(int(predictions[ran])))]).upper())

print("Actual Value: "+str(key_list[val_list.index(y_test[ran])]).upper())
