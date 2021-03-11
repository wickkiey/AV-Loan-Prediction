import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
                       

# Feature Engineering 

#Gender 
train.Gender.fillna(train.Gender.mode()[0],inplace=True)
test.Gender.fillna(test.Gender.mode()[0],inplace=True)

# Married 
train.Married.fillna(train.Married.mode()[0],inplace=True)
test.Married.fillna(test.Married.mode()[0],inplace=True)


# Dependents
# train.Dependents.fillna(train.Dependents.median(),inplace=True)
# test.Dependents.fillna(train.Dependents.median(),inplace=True)

# Self Employed
train.Self_Employed.fillna(train.Self_Employed.mode()[0],inplace=True)
test.Self_Employed.fillna(test.Self_Employed.mode()[0],inplace=True)

#LoanAmount
train.LoanAmount.fillna(train.LoanAmount.median(),inplace=True)
test.LoanAmount.fillna(train.LoanAmount.median(),inplace=True)

#Loan_Amount_Term     
train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.median(),inplace=True)
test.Loan_Amount_Term.fillna(train.Loan_Amount_Term.median(),inplace=True)


#Loan_Amount_Term     
train.Credit_History.fillna(train.Credit_History.mode()[0],inplace=True)
test.Credit_History.fillna(train.Credit_History.mode()[0],inplace=True)


cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']

X = train.drop(['Loan_ID','Loan_Status'],axis=1)
y = train['Loan_Status']
y_enc = y.map({'Y':1,'N':0})
cat_enc = ce.CatBoostEncoder(cols=cat_cols)


lr_pipe = Pipeline(steps=[
        ('enc',cat_enc),
        ('scl',StandardScaler()),
        ('mdl',LogisticRegression(class_weight='balanced'))
])

rf_pipe = Pipeline(steps=[
        ('enc',cat_enc),
        ('scl',StandardScaler()),
        ('mdl',RandomForestClassifier(n_estimators=400,class_weight='balanced'))
])

cv_score = cross_val_score(lr_pipe,X,y_enc,scoring='accuracy')

print("CV Score " ,cv_score, cv_score.mean(), cv_score.std())



# Model prediction
# model = lr_pipe
# model.fit(X,y_enc)
# y_pred = model.predict(test[X.columns])
# res = pd.DataFrame()
# res['Loan_ID'] = test.Loan_ID
# res['Loan_Status'] = y_pred
# res['Loan_Status'] = res['Loan_Status'].map({1:'Y',0:'N'})

# res.to_csv("lr_pipe.csv",index=False)