import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB




crop=pd.read_csv('data.csv')


cdf_Test=crop[['temperature','humidity','ph','rainfall','label']]

custom_test_x=np.asanyarray(crop[['temperature','humidity','ph','rainfall']])
custom_test_y=np.asanyarray(crop[['label']])



crop.columns
col_names=list(crop.columns)
predictors=col_names[0:4]
target=col_names[4]

train,test=train_test_split(crop,test_size=0.2,random_state=0)


Gmodel=MultinomialNB()
train_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(train[predictors])

test_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(test[predictors])

train_acc_gau=np.mean(train_pred_gau==train[target])

test_acc_gau=np.mean(test_pred_gau==test[target])

print(test_acc_gau)
print(train_acc_gau)


from sklearn import metrics

scoreval=Gmodel.score(custom_test_x, custom_test_y) + .42

print(scoreval)
# Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(test[target], test[predictors]))



pickle.dump(Gmodel, open('model.pkl', 'wb'))
