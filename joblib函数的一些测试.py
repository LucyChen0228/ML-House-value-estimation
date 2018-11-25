from sklearn import svm
from sklearn.externals import joblib


X = [[0, 0], [1, 1]]
y = [0, 1]
test = svm.SVC()
test.fit(X,y)
joblib.dump(test,'model_test.m')

