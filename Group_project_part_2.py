import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv('F:/MSFE/machine_learning/GP/MLF_GP2_EconCycle.csv')
print(data.head())
#delete the date in the data_column
data.set_index('Date', inplace=True)
#EDA
print('Number of rows of data: ', data.shape[0])
print('Number of columns of data: ', data.shape[1])
print(data.info())
print(data.describe())

data['USPHCI'].plot()
plt.title('Trend of USPHCI')
plt.ylabel('USPHCI')
plt.show()

data[['PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD']].plot()
plt.title('Trend of the PCTs')
plt.ylabel('PCT')
plt.show()

# Correlation
corMat = pd.DataFrame(data.corr())
print(corMat)

sns.set(style='whitegrid', context='notebook')
cm = np.corrcoef(data.values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm,
                 linewidths = 0.1,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.1f',
                 annot_kws={'size': 8},
                 yticklabels=data.columns,
                 xticklabels=data.columns)
plt.show()

sns.pairplot(data[['T1Y Index', 'T2Y Index',
                   'CP1M', 'CP3M', 'CP1M_T1Y',
                   'CP3M_T1Y', 'USPHCI']])
plt.show()

# Q-Q Plot
stats.probplot(data.USPHCI, dist="norm", plot=plt)
plt.show()

#Regression
# Drop the USPHCI column
data.drop('USPHCI', axis=1, inplace=True)
print(data.head())

# Split X and y
X = data[data.columns[:-3]]
print(X.shape)
y1 = data['PCT 3MO FWD']
print(y1.shape)
y2 = data['PCT 6MO FWD']
print(y2.shape)
y3 = data['PCT 9MO FWD']
print(y3.shape)

# Z-Score Normalization
normalized_X = X.apply(lambda x : (x-np.mean(x))/(np.std(x)))
normalized_X.describe()

# PCA
pca = PCA(n_components=2)
pca.fit(normalized_X)
pca_X = pca.transform(normalized_X)
print(pca_X.shape)

#Linear Regression on PCT 3MO FWD
# Split data after PCA into training and test sets 7:3
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(pca_X, y1, test_size=0.3, random_state=1)
print('Shape of X_train: ', X_train_1.shape)
print('Shape of y_train: ', y_train_1.shape)
print('Shape of X_test: ', X_test_1.shape)
print('Shape of y_test: ', y_test_1.shape)

# Linear Regression for 'PCT 3MO FWD'
model_1 = LinearRegression().fit(X_train_1, y_train_1)
print('coefficient: ', model_1.coef_)
print('intercept: ', model_1.intercept_)
print('training set R2: ', model_1.score(X_train_1, y_train_1))
print('training set MSE: ', mean_squared_error(model_1.predict(X_train_1), y_train_1))
print('testing set R2: ', model_1.score(X_test_1, y_test_1))
print('testing set MSE: ', mean_squared_error(model_1.predict(X_test_1), y_test_1))

plt.plot(model_1.predict(X_test_1))
plt.plot(y_test_1)
plt.title('Linear Regression Result on Test Set: PCT 3MO FWD')
plt.legend(['Predict', 'Real'])
plt.show()

#Decision Tree Regression on PCT 6MO FWD
# Split data after PCA into training and test sets 7:3
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(pca_X, y2, test_size=0.3, random_state=2)
print('Shape of X_train: ', X_train_2.shape)
print('Shape of y_train: ', y_train_2.shape)
print('Shape of X_test: ', X_test_2.shape)
print('Shape of y_test: ', y_test_2.shape)

model_2 = DecisionTreeRegressor()
param = {'max_depth':[2, 5, 8, 10]}
gs = GridSearchCV(model_2, param, cv=10)
gs.fit(X_train_2, y_train_2)

y_pred = gs.predict(X_test_2)
r2 = gs.score(X_test_2, y_test_2)
mse = mean_squared_error(y_pred, y_test_2)

print('10-fold Cross Validation: ')
print("Tuned Decision Tree Parameter: {}".format(gs.best_params_))
print("Tuned Decision Tree R2: {}".format(r2))
print("Tuned Decision Tree MSE: {}".format(mse))

plt.plot(y_pred)
plt.plot(y_test_2)
plt.title('Decision Tree Regression Result on Test Set: PCT 6MO FWD')
plt.legend(['Predict', 'Real'])
plt.show()

#Ridge Regression on PCT 3MO 9WD
# Split normalized data (without PCA because Ridge can select important features) into train set and test set 7:3
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(normalized_X, y3, test_size=0.3, random_state=3)
print('Shape of X_train: ', X_train_3.shape)
print('Shape of y_train: ', y_train_3.shape)
print('Shape of X_test: ', X_test_3.shape)
print('Shape of y_test: ', y_test_3.shape)

model_3 = Ridge()
param = {'alpha':[0.001, 0.01, 0.1, 1, 10, 15, 20]}
gs = GridSearchCV(model_3, param, cv=10)
gs.fit(X_train_3, y_train_3)

y_pred = gs.predict(X_test_3)
r2 = gs.score(X_test_3, y_test_3)
mse = mean_squared_error(y_pred, y_test_3)

print('10-fold Cross Validation: ')
print("Tuned Ridge Parameter: {}".format(gs.best_params_))
print("Tuned Ridge R2: {}".format(r2))
print("Tuned Ridge MSE: {}".format(mse))

plt.plot(y_pred)
plt.plot(y_test_3)
plt.title('Ridge Regression Result on Test Set: PCT 9MO FWD')
plt.legend(['Predict', 'Real'])
plt.show()


#Ensembling by using GradientBoostingRegressor
import time
N = {'n_estimators':[50,100,200,300,400,500,600]}
from sklearn.ensemble import GradientBoostingRegressor
model_4 = GradientBoostingRegressor(max_depth=4,random_state=16)
gb = GridSearchCV(model_4,N,cv=10)
gb.fit(X_train_3,y_train_3)
y_pred = gb.predict(X_test_3)
r2 = gb.score(X_test_3, y_test_3)
mse = mean_squared_error(y_pred, y_test_3)

print('Ensemble by GradientBoostingRegressor: ')
print("GradientBoostingRegressor Parameter: {}".format(gb.best_params_))
print("GradientBoostingRegressor R2: {}".format(r2))
print("GradientBoostingRegressor MSE: {}".format(mse))

plt.plot(y_pred)
plt.plot(y_test_3)
plt.title('GradientBoostingRegressoe Result on Test Set: PCT 9MO FWD')
plt.legend(['Predict', 'Real'])
plt.show()