# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"""
#Fix missing Datar̥
from sklearn.impute import SimpleImputer 

imputer=SimpleImputer()
X[:,1:3]= imputer.fit_transform(X[:,1:3])
"""

"""
#Encoding categorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

ctX = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ctX.fit_transform(X),dtype=np.int)
label_encoderY=LabelEncoder()
y=label_encoderY.fit_transform(y)
"""

#Splitting in test and training
from sklearn.model_selection import train_test_split as TTS

X_train,X_test,y_train,y_test= TTS(X,y, test_size=0.2,random_state=0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(X_train)
X_train=ss.transform(X_train)
X_test=ss.transform(X_test)
"""