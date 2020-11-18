import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle



#Import Data
df=pd.read_csv('D:\Github\ML-Health-Status-Detection-Web-App\pythonProject2\Heart.csv')

#Dropping irrelevent columns and missing values
df.drop(['education'],axis=1,inplace=True)
df.dropna(axis=0,inplace=True)

# Resampling- Upsampling
minority = df[df.TenYearCHD==1]
majority = df[df.TenYearCHD==0]
minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])
df = pd.concat([minority_upsample, majority], axis=0)

# Splitting the data
from sklearn.model_selection import train_test_split
X =  df.drop('TenYearCHD', axis=1)
Y =  df['TenYearCHD']
X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=Y)
#Model
from xgboost import XGBClassifier

model = XGBClassifier(random_state=123)
model.fit(X_train, y_train)

filename = 'heart.sav'
pickle.dump(model, open(filename, 'wb'))



