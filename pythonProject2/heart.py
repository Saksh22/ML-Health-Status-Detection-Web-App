import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle



#Import Data
df=pd.read_csv('D:\Github\ML-Health-Status-Detection-Web-App\pythonProject2\heart1.csv')


#Preprocess Data
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)

df = df.drop(columns = ['cp', 'thal', 'slope'])
y = df.target.values
x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)

filename = 'heart.sav'
pickle.dump(rf, open(filename, 'wb'))



