import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.cross_validation import train_test_split
import numpy as np
from __future__ import division
import math
from sklearn.cross_validation import KFold
import xgboost as xgb

def myRMSPE(yhat,y):
    yhat=np.array(yhat)
    y=np.array(y) 
    index = np.where(y!=0)
    yhat=yhat[index]
    y=y[index]
    return math.sqrt(np.mean(((y-yhat)/y)**2))
  


def myRMSPE_xg(yhat,y):
    
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    r=myRMSPE(yhat,y)
    
    return "rmspe", r
    
train=pd.read_csv(r'/home/zyybear/Desktop/UNH/RS_Kaggle/train/train_FE0.csv')
test=pd.read_csv('/home/zyybear/Desktop/UNH/RS_Kaggle/test/test_FE0.csv')   

train.drop(['Store','Customers','Date','WeekOfYear'],axis=1,inplace=True)
test.drop(['Id','Date','Store','WeekOfYear'],axis=1,inplace=True)
myt=test.sort(axis=1)

sub_train,sub_test=train_test_split(train, test_size=0.0012)

X=sub_train.drop('Sales',axis=1)
y=sub_train['Sales']
y=np.log1p(y)
X=X.sort(axis=1)

x_val=sub_test.drop('Sales',axis=1)
x_val=x_val.sort(axis=1)
y_val=sub_test['Sales']
y_val=np.log1p(y_val)

x_val=np.array(x_val)
y_val=np.array(y_val)

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.05,
          "max_depth": 9,
          "subsample": 0.95,
          "colsample_bytree": 0.7,
          "silent": 1,
          #"seed": 1301
          }

num_trees=4000


xgtrain = xgb.DMatrix(X, y)
xgval = xgb.DMatrix(x_val, y_val)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]

XGB_model = xgb.train(params, xgtrain,num_trees,feval=myRMSPE_xg,
                     evals=watchlist,early_stopping_rounds=100,
                     verbose_eval=True)
preds = XGB_model.predict(xgb.DMatrix(myt))
preds =np.expm1(preds)
test=pd.read_csv(r'/home/zyybear/Desktop/UNH/RS_Kaggle/test/test.csv')
result = pd.DataFrame({'Id': test.Id})
result['Sales']=preds
result.to_csv('XGB_12-13_5.csv', index=False, sep=',')
