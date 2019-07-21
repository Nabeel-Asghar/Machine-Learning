import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

x = preprocessing.LabelEncoder()
buying = x.fit_transform(list(data["buying"]))
maint = x.fit_transform(list(data["maint"]))
doors = x.fit_transform(list(data["door"]))
persons = x.fit_transform(list(data["persons"]))
lug = x.fit_transform(list(data["lug_boot"]))
safety = x.fit_transform(list(data["safety"]))
cls = x.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, doors, persons, lug, safety))
y = list(cls)

x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)
predicted = model.predict((x_test))

names = ["unacc", "bad", "good", "excellent"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors(([x_test[x]]), 9 , True)
    print("N:", n)


