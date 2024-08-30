# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SENTHIL KUMARAN C
RegisterNumber:  21223220103

CODE:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:

df.head()
![image](https://github.com/user-attachments/assets/6a74d6dc-7950-4d7a-a97d-242855f4790e)
df.tail()
![image](https://github.com/user-attachments/assets/f7dfa51b-35c3-4109-87e0-0832b76aa888)
Array value of X
![image](https://github.com/user-attachments/assets/bf5ef36d-4297-45b5-b6ce-845ca1cfeb15)
Array value of Y
![image](https://github.com/user-attachments/assets/395483dd-9ace-473b-88cb-de26227b5e0a)
Array values of Y test
![image](https://github.com/user-attachments/assets/35c0b8ba-20fa-4edd-b9a8-203460d43d33)
Training Set Graph
![image](https://github.com/user-attachments/assets/6447374a-586e-4bf5-8665-96135eee025c)
Test Set Graph
![image](https://github.com/user-attachments/assets/661ebe7c-2031-43f4-aa64-56f7535371a4)
Values of MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/7de1184a-7f50-42b6-95d0-4dfcfe6543c8)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
