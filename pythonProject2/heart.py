import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
#import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# %matplotlib inline
import streamlit as st
import joblib
import pickle



#Import Data
df=pd.read_csv('C:/Users/saksh/PycharmProjects/pythonProject2/Heart.csv')


#Preprocess Data
df.drop(['education'],axis=1,inplace=True)
df.rename(columns={'male':'Sex_male'},inplace=True)
df.dropna(axis=0,inplace=True)

st.subheader('Data Information:')
#Display Dataset
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)

from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(df)
heart_df_constant.head()

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df_constant.columns[:-1]
model=sm.Logit(df.TenYearCHD,heart_df_constant[cols])
result=model.fit()

def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(heart_df_constant,df.TenYearCHD,cols)
params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

import sklearn
new_features=df[['age','Sex_male','cigsPerDay','totChol','sysBP','BMI','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)

#Model
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

filename = 'heart.sav'
pickle.dump(model, open(filename, 'wb'))



