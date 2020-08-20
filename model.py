import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("http://bit.ly/w-data")

X=data.iloc[:,:-1]
y=data.iloc[:,1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

pickle.dump(model,open('model.pkl','wb'))