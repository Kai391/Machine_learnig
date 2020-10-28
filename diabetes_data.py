import matplotlib.pyplot as plt
import numpy as ny
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

data = datasets.load_diabetes()
# print(data.data)
data_x = data.data[:,ny.newaxis,2]
# data_x = data.data
# print(data_x)
data_x_train= data_x[:-30]
# print(data_x_train)
data_x_test= data_x[-20:]

data_y_train= data.target[:-30]
data_y_test= data.target[-20:]

model= linear_model.LinearRegression()
model.fit(data_x_train,data_y_train)
data_predict=model.predict(data_x_test)

print("means square error: ",mean_squared_error(data_y_test,data_predict))
print('weights: ',model.coef_)
print('intercepts: ',model.intercept_)

plt.scatter(data_x_test,data_y_test)
plt.plot(data_x_test,data_predict)
plt.show()
# print(data_x_test,data_y_test)