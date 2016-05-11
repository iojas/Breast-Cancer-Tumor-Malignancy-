from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

x=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
            header=None,names=['BI-RADS','Age','Shape','Margin','Density','Severity'],na_values='?')
x=x.dropna()

feature_cols=['BI-RADS','Age','Shape','Margin','Density']
a=x[feature_cols]
b=x['Severity']


a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.25)

knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(a_train,b_train)
ans=knn.predict(a_test)
print "Accuracy of k nearest neighbors:  ",metrics.accuracy_score(ans,b_test)

ll=LogisticRegression()
ll.fit(a_train, b_train)
ans=ll.predict(a_test)

print "Accuracy of Logistic regression: ",metrics.accuracy_score(b_test,ans)
