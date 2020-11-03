#Importing Libraries
import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample
import pickle

#Data Loading
df=pd.read_csv('D:/Github/ML-Health-Status-Detection-Web-App/pythonProject2/diabetes.csv')
df_name=df.columns

#Replacing null values
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)


#Resampling-UpSampling
minority = df[df.Outcome==1]
majority = df[df.Outcome==0]

minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])
df = pd.concat([minority_upsample, majority], axis=0)

#Data Spliting
X=df[df_name[0:8]]
Y=df[df_name[8]]


#Split data into train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0,stratify=Y)



model = ExtraTreesClassifier(random_state=123)
model.fit(X_train, Y_train)


filename = 'diabetes.sav'
pickle.dump(model, open(filename, 'wb'))