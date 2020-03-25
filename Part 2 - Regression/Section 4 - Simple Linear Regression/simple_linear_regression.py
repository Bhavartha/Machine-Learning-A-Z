import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#Splitting in test and training
from sklearn.model_selection import train_test_split as TTS

X_train,X_test,y_train,y_test= TTS(X,y, test_size=1/3,random_state=0)


# Simple linear Regression
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred= regressor.predict(X_test)


# Plotting
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.plot(X_train, y_train, color='green')
plt.title('Salary - Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()