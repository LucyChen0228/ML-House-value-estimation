import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

'''from sklearn.externals import joblib'''

# Load the data set
df = pd.read_csv("ml_house_data_set_updated.csv")


'''删去无用的变量'''
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']


features_df = pd.get_dummies(df, columns=['garage_type', 'city'])
''''pd_get_dummies 将离散特征取值之间没有大小关系的，转为one-hot 编码(机器学习中常用的二进制编码，进行二进制操作，columns指定需要实现类别转换的列名'''

'''因为预测的是sale_price， 故先删除'''
del features_df['sale_price']


X = features_df.as_matrix()
y = df['sale_price'].as_matrix()


#将数据集和训练集分开
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.3,random_state=0)



'''gradient boosting算法'''
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train )
joblib.dump(model, 'trained_house_model_update.pkl')

mse=mean_absolute_error(y_train,model.predict(X_train))
print('traning set mean absolute error:%.4f' %mse)
print(model.predict(X_train))


mse=mean_absolute_error(y_test,model.predict(X_test))
print('test set mean absolute error:%.4f' %mse)
print(model.predict(X_test))
'''可以使用同样的mse ,因为顺序不同，按照顺序的先后来进行改变变量的赋值'''

