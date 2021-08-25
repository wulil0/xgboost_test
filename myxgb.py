import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import xgboost as xgb

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('test2.csv')
df.columns = [
    'day', 'country', 'state', 'lat', 'long', 'confirmed', 'recovered',
    'deaths'
]
x = df[['confirmed', 'recovered']][100:300]
y = df[['deaths']][100:300]

# x_train = df[['confirmed', 'recovered']][100:260]
# x_test = df[['confirmed', 'recovered']][260:300]

# y_train = df[['deaths']][100:260]
# y_test = df[['deaths']][260:300]
# print(y_train)

# lbl = preprocessing.LabelEncoder()
# x['day'] = lbl.fit_transform(x['day'].astype(str))  #将提示的包含错误数据类型这一列进行转换
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=14)
# print(y_test)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test,label=y_test)
# 1. 参数构建
params = {
    'max_depth': 10,
    'eta': 1,
    'silent': 1,
    'objective': 'reg:squarederror',
    # 'max_depth': 5
}
num_round = 2
# 2. 模型训练
bst = xgb.train(params, dtrain, num_round)
# 3. 模型保存
bst.save_model('xgb.model')

# 模型预测
y_pred = bst.predict(dtest)
# print(mean_squared_error(y_pred, y_test))
print(y_pred)

# 4. 加载模型
bst2 = xgb.Booster()
bst2.load_model('xgb.model')
# 5 使用加载模型预测
y_pred2 = bst2.predict(dtest)
print(mean_squared_error(y_pred2, y_test))

plt.figure(figsize=(12, 6), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'True Value')
plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'XGBoost Predict')
plt.xlabel(u'time')
plt.ylabel(u'number')
plt.legend(loc='lower right')
plt.grid(True)
plt.title(u'covid-19 predict')
plt.show()