import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from feature_engine.selection import DropCorrelatedFeatures,SmartCorrelatedSelection
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import pickle

import warnings
warnings.filterwarnings(action='ignore')

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', names = col_names, delim_whitespace=True)


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

sel_v = VarianceThreshold(threshold=0.01)
sel_v.fit(x_train)

x_train = sel_v.transform(x_train)
x_test = sel_v.transform(x_test)

drop_dup = DropCorrelatedFeatures(threshold=1.0, missing_values='raise')

drop_dup.fit(x_train)
drop_dup.get_feature_names_out()

dup = drop_dup.transform(x_train)

rfr = RandomForestRegressor()
drop_cor = SmartCorrelatedSelection(estimator=rfr)


drop_cor.fit(x_train, y_train)
x_train = drop_cor.transform(x_train)

x_test = drop_cor.transform(x_test)
sel_per = SelectPercentile(percentile=95)
sel_per.fit(x_train,y_train)
col = sel_per.transform(x_train)

x_train = sel_per.transform(x_train)
x_test = sel_per.transform(x_test)


print(x_train.shape, x_test.shape)
model = XGBRegressor(base_score=0.6, max_depth=21, subsample=0.9)

model.fit(x_train, y_train)

xgb_y_pred = model.predict(x_test)

print(mean_squared_error(xgb_y_pred, y_test))
print(r2_score(xgb_y_pred, y_test))
print(mean_absolute_error(xgb_y_pred, y_test))


pickle.dump(model, open('model.pkl', 'wb'))


inp = [[    0.00632,  18, 2.31, 0.538, 6.575, 65.2, 4.09, 1, 15.3, 396.9, 4.98]]
print(model.predict(inp))
#print(model.predict(std_.transform([[0.02729, 0, 7.07, 0.469, 7.185, 61.1, 4.9671, 2, 17.8, 392.83, 4.03]])))
print(inp)