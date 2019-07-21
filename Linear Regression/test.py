# import tensorflow
# import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3" #label - what you are trying to get

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

best = 0
# for _ in range(20):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: " + str(acc))
#
#     if acc > best:
#         best = acc
#         with open("studentgrades.pickle", "wb") as f:
#             pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


print(linear.coef_)
print(linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

