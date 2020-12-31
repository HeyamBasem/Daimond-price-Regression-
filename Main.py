import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = r'D:\diamond project\diamonds.csv'
dm = pd.read_csv(x)

print(dm.info())
print(dm.head())
print(dm.describe())

print(dm.isnull().sum())
#thankfully, no null values

dm.hist(bins=50, figsize=(20,15))
plt.show()

#cut and price
sns.set(style="darkgrid")
sns.boxplot(x="cut", y="price", data=dm)
plt.show()

#the count of diamonds based on clarity
sns.countplot(x='clarity',data=dm , )
plt.show()

#color and price relation
sns.barplot(x="color", y="price", data=dm)
plt.show()


#color , cut and price all together
g = sns.FacetGrid(dm, col="cut", height=5, aspect=.5 )
g.map(sns.barplot, "color", "price")
plt.show()

#showing the correlation between factors
sns.heatmap(dm.corr(),annot=True)
plt.show()





cut = pd.get_dummies(dm['cut'],drop_first=True)
color = pd.get_dummies(dm['color'],drop_first=True)
clarity= pd.get_dummies(dm['clarity'],drop_first=True)
dm.drop(['cut','color','clarity','Unnamed: 0'],axis=1,inplace=True)
dm = pd.concat([dm,cut,color,clarity],axis=1)
print(dm.head())
from sklearn.model_selection import train_test_split
x = dm[['carat', 'depth', 'table', 'x', 'y', 'z', 'Good', 'Ideal','Premium', 'Very Good', 'E', 'F', 'G', 'H', 'I', 'J', 'IF', 'SI1',
       'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']]
y = dm['price']


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score

def evaluation(mod ,X_test,y_test,predictions) :
       plt.scatter(y_test, predictions)
       plt.show()
       sns.distplot((y_test - predictions), bins=50)
       plt.show()
       print('MAE:', metrics.mean_absolute_error(y_test, predictions))
       print('MSE:', metrics.mean_squared_error(y_test, predictions))
       print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
       acc = mod.score(X_test, y_test) * 100
       print("accuracy: ", acc.round(2), "%")
       r2 = r2_score(y_test, predictions) * 100
       print('R Squared: ', r2.round(2), "%")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
evaluation(lm,X_test,y_test,predictions)


tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(X_train,y_train)
predictions = tree_reg.predict(X_test)
evaluation(tree_reg ,X_test,y_test,predictions)

fo_reg = RandomForestRegressor(n_estimators = 15 , random_state = 50)
fo_reg.fit(X_train,y_train)
predictions = fo_reg.predict(X_test)
evaluation(fo_reg  , X_test,y_test,predictions)

