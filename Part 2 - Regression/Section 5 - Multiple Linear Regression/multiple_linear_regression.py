# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#Categorial data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ctX = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ctX.fit_transform(X),dtype=np.float)[:,1:]


#Splitting in test and training
from sklearn.model_selection import train_test_split as TTS
X_train,X_test,y_train,y_test= TTS(X,y, test_size=0.2,random_state=0)


#Train
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred= regressor.predict(X_test)


#Backward elimination
import statsmodels.api as sm
X= np.append(arr= np.ones((50,1),np.float),values=X,axis=1)

X_optimal= X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(y,X_optimal).fit()
regressor_OLS.summary()

X_optimal= X[:,[0,1,3,4,5]]
regressor_OLS= sm.OLS(y,X_optimal).fit()
regressor_OLS.summary()

X_optimal= X[:,[0,3,4,5]]
regressor_OLS= sm.OLS(y,X_optimal).fit()
regressor_OLS.summary()

X_optimal= X[:,[0,3,5]]
regressor_OLS= sm.OLS(y,X_optimal).fit()
regressor_OLS.summary()

X_optimal= X[:,[0,3]]
regressor_OLS= sm.OLS(y,X_optimal).fit()
regressor_OLS.summary()



#Splitting in test and training
from sklearn.model_selection import train_test_split as TTS
X_train,X_test,y_train,y_test= TTS(X_optimal,y, test_size=0.2,random_state=0)


#Train
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred2= regressor.predict(X_test)