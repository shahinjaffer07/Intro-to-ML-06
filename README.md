# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and Load the Dataset.

2.Drop Irrelevant Columns (sl_no, salary).

3.Convert Categorical Columns to Category Data Type.

4.Encode Categorical Columns as Numeric Codes.

5.Split Dataset into Features (X) and Target (Y).

6.Initialize Model Parameters (theta) Randomly.

7.Define Sigmoid Activation Function.

8.Define Logistic Loss Function (Binary Cross-Entropy).

9.Implement Gradient Descent to Minimize Loss.

10.Train the Model by Updating theta Iteratively.

11.Define Prediction Function Using Threshold (0.5).

12.Predict Outcomes for Training Set.

13.Calculate and Display Accuracy.

14.Make Predictions on New Data Samples.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHAHIN J
RegisterNumber: 212223040190
*/
```
```
import pandas as pd
import numpy as np
df=pd.read_csv("Placement_Data.csv")
df
df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
## display dependent variables
y
theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-Y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
print(y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:
# Placement Dataset

![image](https://github.com/user-attachments/assets/c4a576ab-96b3-4d3b-9d96-cb0854ceca32)

# Dataset after Feature Engineering

![image](https://github.com/user-attachments/assets/d16a7b0d-0786-42c7-a0a8-bfe16f82b6c4)

# Datatypes of Feature column

![image](https://github.com/user-attachments/assets/1013440a-9230-44cf-8357-5fd2c69311d9)

# Dataset after Encoding

![image](https://github.com/user-attachments/assets/99b7c543-eadc-49e7-afcb-6266cf6959ca)

# Y Values

![image](https://github.com/user-attachments/assets/ad2dc197-906d-400d-ae8f-1e2005ea33d3)

# Accuracy

![image](https://github.com/user-attachments/assets/6c5a627a-f293-46f1-b8b4-38b4dee86d6a)

# Y Predicted

![image](https://github.com/user-attachments/assets/b327f3e7-9b56-4cc7-b728-6eae94fb8bec)

# Y Values

![image](https://github.com/user-attachments/assets/06a9ca30-c450-4e5c-a1fb-cb2b4ae06ea6)

# Y Predicted with different X Values

![image](https://github.com/user-attachments/assets/6ccd3912-23d1-4370-a61b-ed57fba61a77)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

