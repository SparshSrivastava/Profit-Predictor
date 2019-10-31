# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Data Set
dataset = pd.read_excel('/home/sparsh/Documents/Profit Predictor Project/Linear Regression/Book2.xlsx')
X = dataset.iloc[:,3].values
Y = dataset.iloc[:,4].values

# Splitting the Data into Training and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/6,random_state=0)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test)

# Applying Linear Regression to training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test Set
Y_pred = regressor.predict(X_test)

# Visualising the Training Set
plt.scatter(X_train,Y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Analysing Variation of Monthly Income of Riders Vs Population of the city(Traiing Set)')
plt.xlabel('Population of the city')
plt.ylabel('Monthly income of the Riders')
plt.show()

# Visualisig the Test Set
plt.scatter(X_test,Y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Analysing Variation of Monthly Income of Riders Vs Population of the city (Test Set)')
plt.xlabel('Population of the city')
plt.ylabel('Monthly Income of the Riders')
plt.show()

from sklearn.metrics import r2_score
r_squared=r2_score(Y_test,Y_pred)
print(r_squared)