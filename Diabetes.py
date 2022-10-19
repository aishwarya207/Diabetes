import pandas as pd
data=pd.read_csv("dia.csv")
data.head(10)
from sklearn.naive_bayes import gaussianAB
model=gaussianAB()
model.fit(p_train,q_train)
q_pred=model.predict(p_train)
from sklearn import metrices
print("accuracy:",metrices.accuracy_score(q_test,q_pred))
test_pred=model.predict(p_test)
print(metrices.classificaion_report(q_test,test_pred))
print(metrices.confusion_matrix(q_test,test_pred))