#this is a baseline submission.
#no real feature engineering and tuned model

from __future__ import division
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.cross_validation import train_test_split
import numpy as np


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def RMSPE(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

#read in training and test data set
folder = 'data/'
real_train=pd.read_csv(folder + 'train.csv')
real_test=pd.read_csv(folder + 'test.csv')

#check the missing value in real_test table
print real_test.isnull().sum()

#there are some missing value in Open columns, at this point we just simply fill in with 1
real_test.Open.fillna(1, inplace=True)

#check missing value again,supposed to no missing value
print real_test.isnull().sum()

#check the column type 
print real_test.dtypes

#convert float type to int for less trouble in future
real_test['Open']=real_test['Open'].astype(int)

#check type again,should not have any float type value
print real_test.dtypes

#feature enginnering, very basic at this level
real_train['Month'] = pd.DatetimeIndex(real_train['Date']).month
real_test['Month']=pd.DatetimeIndex(real_test['Date']).month

#convert some columns type to object for dummies
for col in ['DayOfWeek','Month','Promo','SchoolHoliday','Open','StateHoliday']:
    real_train[col]=real_train[col].astype(str)
    real_test[col]=real_test[col].astype(str)

#check the type
print real_test.dtypes


#get dummy table for train ready
categorical_variables=['DayOfWeek','Month','Promo','SchoolHoliday','Open','StateHoliday']
regular_variables=['Store','Date','Sales','Customers']

dummy_table=pd.DataFrame()
for var in categorical_variables:
    dummy_table=pd.concat([dummy_table,pd.get_dummies(real_train[var], prefix=var)], axis=1) 
    
real_train=pd.concat([dummy_table,real_train[regular_variables]],axis=1)


#get dummy table for test ready
categorical_variables=['DayOfWeek','Month','Promo','SchoolHoliday','Open','StateHoliday']
regular_variables=['Store','Date']

dummy_table=pd.DataFrame()
for var in categorical_variables:
    dummy_table=pd.concat([dummy_table,pd.get_dummies(real_test[var], prefix=var)], axis=1) 
    
real_test=pd.concat([dummy_table,real_test[regular_variables]],axis=1)

# add some dummies columns to real_test table to make sure the size
# is consistant with training table
for varMonth in [1,2,3,4,5,6,7,10,11,12]:
    real_test['Month_'+str(varMonth)]=0

for varhol in ['b','c']:
    real_test['StateHoliday_'+varhol]=0

#train RF model
X=real_train.drop(['Store','Sales','Customers','Date'],axis=1)
y=real_train['Sales']
RF_model=RF(n_jobs=-1)
RF_model.fit(X,y)

#create submission file
myt=real_test.drop(['Store','Date'],axis=1)
preds=RF_model.predict(myt)

test=pd.read_csv(folder + 'test.csv')
result = pd.DataFrame({'Id': test.Id})
result['Sales']=preds
result.to_csv('baseline10-29.csv', index=False, sep=',')
