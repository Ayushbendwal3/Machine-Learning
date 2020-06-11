import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model


wine_set = pd.read_csv(
    "Linear Regression/Datasets/winequality-red.csv", sep=";")

recode = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}

wine_set["quality_num"] = wine_set["quality"].map(recode)


def Log_regression(wine_set):
    x = wine_set[["sulphates", "alcohol"]]
    y = wine_set[["quality_num"]]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.2)

    model = linear_model.LogisticRegression()
    model = model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Confusion Matrix:",
          sklearn.metrics.confusion_matrix(y_test, predictions))
    print("Accuracy:", sklearn.metrics.accuracy_score(y_test, predictions))


Log_regression(wine_set)
