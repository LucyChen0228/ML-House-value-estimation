import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

'''from sklearn.externals import joblib'''

# Load the data set
df = pd.read_csv("house_data_set.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']
'''del df 的意思是什么,删去无用的变量'''

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])
''''pd_get_dummies 将离散特征取值之间没有大小关系的，转为one-hot 编码(机器学习中常用的二进制编码，进行二进制操作，columns指定需要实现类别转换的列名'''

# Remove the sale price from the feature data
del features_df['sale_price']

'''问题：sale_price 并不属于feature_df 类比里面？'''
'''如果不转化为one-hot 编码会怎么样？'''

# Create the X and y arrays
X = features_df.as_matrix()
y = df['sale_price'].as_matrix()


#将数据集和训练集分开
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
'''train_test_split 函数：交叉验证'''
'''语法：train_data：所要划分的样本特征集

train_target：所要划分的样本结果

test_size：样本占比，如果是整数的话就是样本的数量

random_state：是随机数的种子。'''


model = ensemble.GradientBoostingRegressor()
'''ensemble 随机'''


param_grid={
    'n_estimators':[500,1000,3000],
    'learning_rate':[0.1,0.05,0.02,0.01],
    'max_depth':[4,6],
    'loss':['huber','ls','lad'],
    'min_samples_leaf':[3,5,7,9],
    'max_features':[0.1,0.3,1.0]
}

'''用grid 去调整超参数达到最佳'''
'''loss: 选择损失函数，默认值为ls(least squres)

learning_rate: 学习率，模型是0.1

n_estimators: 弱学习器的数目，默认值100，decision tree 的数量，越高越好，但是越高的话，会使运算速度变慢

max_depth: 每一个学习器的最大深度，限制回归树的节点数目，默认为3，决策树的

min_samples_split: 可以划分为内部节点的最小样本数，默认为2

min_samples_leaf: 叶节点所需的最小样本数，默认为1'''


# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_model.pkl')

gs_cv= GridSearchCV(model, param_grid, n_jobs=4)
'''gridsearch CV 自动调参数，n_jobs:并行数，int：个数,-1：跟CPU核数一致, 1:默认值'''

gs_cv.fit(X_train,y_train)



print(gs_cv.best_params_)
'''best_params和best_index的区别是什么?'''
'''为什么在这里best_params后面还要加_'''


'''joblib.dump(model,'trained_house_model.pkl')'''

'''pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西'''
'''csv文件：逗号分隔符文件，可以使用excel打开'''

mse=mean_absolute_error(y_train,model.predict(X_train))
print('traning set mean absolute error:%.4f' %mse)
print(model.predict(X_train))


mse=mean_absolute_error(y_test,model.predict(X_test))
print('test set mean absolute error:%.4f' %mse)
print(model.predict(X_test))
'''可以使用同样的mse ,因为顺序不同，按照顺序的先后来进行改变变量的赋值'''


'''GridSearchCV 函数，ensemble 函数，train_test_split 函数'''
