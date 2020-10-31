#Importing Libraries
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

#Data Spliting
X=df[df_name[0:8]]
Y=df[df_name[8]]


#Split data into train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0,stratify=Y)

#Resampling-downSampling
X = pd.concat([X_train, Y_train], axis=1)
not_diabetic = X[X.Outcome==0]
diabetic = X[X.Outcome==1]
from sklearn.utils import resample
not_diabetic_downsampled = resample(not_diabetic,
                          replace=True, # sample with replacement
                          n_samples=len(diabetic), # match number in majority class
                          random_state=27)
downsampled = pd.concat([not_diabetic_downsampled,diabetic])
Y_train = downsampled.Outcome
X_train = downsampled.drop('Outcome', axis=1)

RandomForestClassifier=RandomForestClassifier(random_state = 1,n_estimators= 1000,
 min_samples_split= 5,
 min_samples_leaf=2,
 max_features= 'sqrt',
 max_depth= 10,
 bootstrap= True
 )
RandomForestClassifier.fit(X_train,Y_train)



filename = 'diabetes.sav'
pickle.dump(RandomForestClassifier, open(filename, 'wb'))