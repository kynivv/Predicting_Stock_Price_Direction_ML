from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

plt.style.use('seaborn-darkgrid')

import warnings
warnings.filterwarnings("ignore")


# Data Import
df = pd.read_csv('RELIANCE.csv')
#print(df)


# Data Preparation
df.index = pd.to_datetime(df['Date'])

df = df.drop(['Date'], axis='columns')
#print(df)


df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

X = df[['Open-Close', 'High-Low']]


# Defining Target Variables
Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# Train Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

#print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)


# Model Training and Accuracy
svc = SVC()
cls = svc.fit(X_train,Y_train)

pred_train = cls.predict(X_train)
print(f'Training Accuracy : {1-(mae(Y_train,pred_train))}')

pred_val = cls.predict(X_val)
print(f'Validation Accuracy : {1-(mae(Y_val, pred_val))}')


# Visualization
df['Predicted_Signals'] = cls.predict(X)

df['Return'] = df.Close.pct_change()

df['Strategy_Return'] = df.Return *df.Predicted_Signals.shift(1)

df['Cum_Ret'] = df['Return'].cumsum()

df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.plot(df['Cum_Ret'], color='red')
plt.plot(df['Cum_Strategy'], color='blue')
plt.show()