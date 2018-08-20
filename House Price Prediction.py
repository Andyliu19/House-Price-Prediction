import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold #

data = pd.read_csv("train.csv")

def split_data(df, train_perc = 0.8):

   df['train'] = np.random.rand(len(df)) < train_perc

   train = df[df.train == 1]

   test = df[df.train == 0]

   split_data ={'train': train, 'test': test}

   return split_data

data.head()


quantitative = [col for col in data.columns if data.dtypes[col] != 'object']
quantitative

qualitative = [col for col in data.columns if data.dtypes[col] == 'object']
qualitative

fig = plt.figure(figsize=(14, 14))
sns.heatmap(data[quantitative].corr(), ax=fig.gca())

corr = data[quantitative].corr()["SalePrice"]
corr
corr[abs(corr)>0.4]

selected_quan = corr[abs(corr)>0.4].index
selected_quan

fig = plt.figure(figsize=(14, 14))
sns.heatmap(data[selected_quan].corr(), ax=fig.gca(), annot=True)

selected_quan = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces','GarageArea', 'SalePrice']

data.OverallQual.fillna(data.OverallQual.median(), inplace=True)
data.YearBuilt.fillna(data.YearBuilt.median(), inplace=True)
data.YearRemodAdd.fillna(data.YearRemodAdd.median(), inplace=True)
data.MasVnrArea.fillna(data.MasVnrArea.median(), inplace=True)
data.TotalBsmtSF.fillna(data.TotalBsmtSF.median(), inplace=True)
data["1stFlrSF"].fillna(data["1stFlrSF"].median(), inplace=True)
data.GrLivArea.fillna(data.GrLivArea.median(), inplace=True)
data.FullBath.fillna(data.FullBath.median(), inplace=True)
data.TotRmsAbvGrd.fillna(data.TotRmsAbvGrd.median(), inplace=True)
data.Fireplaces.fillna(data.Fireplaces.median(), inplace=True)
data.GarageYrBlt.fillna(data.GarageYrBlt.median(), inplace=True)
data.GarageCars.fillna(data.GarageCars.median(), inplace=True)
data.GarageArea.fillna(data.GarageArea.median(), inplace=True)
data.SalePrice.fillna(data.SalePrice.median(), inplace=True)

# obtain 80% train 20% test data
split = split_data(data, train_perc=0.8)
train = split['train']
train = train.drop('train', axis = 1)
test = split['test']
test = test.drop('train', axis = 1)


sns.distplot(train["OverallQual"])
sns.distplot(train["YearBuilt"])
sns.distplot(train["YearRemodAdd"])
sns.distplot(train["MasVnrArea"]) ##
sns.distplot(train["TotalBsmtSF"]) ##
sns.distplot(train["FullBath"])
sns.distplot(train["TotRmsAbvGrd"])
sns.distplot(train["Fireplaces"])
sns.distplot(train["GarageArea"]) ##
sns.distplot(train["SalePrice"]) ##

skew_var = ["SalePrice"]
for i in skew_var:
    train[i] = train[i].apply(lambda x: np.log(x))
    test[i] = test[i].apply(lambda x: np.log(x))

#x_train =np.array(train[selected_quan[1:-1]])
#y_train = np.array(train["SalePrice"])
#y_train = y_train.tolist()
#x_train = x_train.tolist()
#
#
#x_test =np.array(test[selected_quan[1:-1]])
#y_test = np.array(test["SalePrice"])
#x_test = x_test.tolist()
#y_test = y_test.tolist()
#
#x_test 

#reg = linear_model.Lasso(alpha = 0.1)
#reg.fit(x,y)
#
#reg.coef_

def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['SalePrice'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data["OverallQual"],y_pred)
        plt.plot(data["OverallQual"],data['SalePrice'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['SalePrice'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

#Initialize predictors to all 15 powers of x
#predictors=['x']
#predictors.extend(['x_%d'%i for i in range(2,16)])

predictors = [selected_quan[0:-1]]
predictors = predictors[0]

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + predictors
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(train, predictors, alpha_lasso[i], models_to_plot)


### ridge coefficient
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['SalePrice'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data["MasVnrArea"],y_pred)
        plt.plot(data["MasVnrArea"],data["SalePrice"],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['SalePrice'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

#Initialize predictors to be set of 15 powers of x
#predictors=['x']
#predictors.extend(['x_%d'%i for i in range(2,16)])


#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + predictors
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(train, predictors, alpha_ridge[i], models_to_plot)

coef_matrix_lasso
coef_matrix_ridge



# using alpha=0.000001, comparison between 3 algorithm on accuracy rate
lassoreg = Lasso(alpha=0.00101,normalize=True, max_iter=1e5)
lassoreg.fit(train[predictors],train['SalePrice'])
y_pred = lassoreg.predict(test[predictors])
lassoreg.score(test[predictors],test['SalePrice'])


ridgereg = Ridge(alpha=0.00000000000001,normalize=True)
ridgereg.fit(train[predictors],train['SalePrice'])
y_pred = ridgereg.predict(test[predictors])
ridgereg.score(test[predictors],test['SalePrice'])

#ridgereg.score(test_org[predictors], test['SalePrice'])

boosting = GradientBoostingRegressor(alpha = 0.00000000000001)
boosting.fit(train[predictors],train['SalePrice'])
y_pred = boosting.predict(test[predictors])
boosting.score(test[predictors],test['SalePrice'])


def mcross_validation(esimator, X, Y, cv=5, random_state=None, need_scaled=True, need_poly=False):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
#        if need_scaled:
#            x_train, x_test = get_scaled_features(x_train, selected_quantitive, x_test)
#    
#        if need_poly:
#            x_train, x_test = get_polyed_features(x_train, selected_quantitive, x_test)

        est =esimator.fit(x_train, y_train)
        y_pred = est.predict(x_test)
        scores.append(est.score(x_test, y_test))
    return scores

scores = mcross_validation(lassoreg, data[predictors], data["SalePrice"], need_poly=True)
'lassoreg', np.mean(scores), scores

scores = mcross_validation(ridgereg, data[predictors], data["SalePrice"], need_poly=True)
'ridgereg', np.mean(scores), scores

scores = mcross_validation(boosting, data[predictors], data["SalePrice"], need_poly=True)
'boosting', np.mean(scores), scores

datatr = pd.read_csv("train.csv")
datat = pd.read_csv("test.csv")

datatr.OverallQual.fillna(datatr.OverallQual.median(), inplace=True)
datatr.YearBuilt.fillna(datatr.YearBuilt.median(), inplace=True)
datatr.YearRemodAdd.fillna(datatr.YearRemodAdd.median(), inplace=True)
datatr.MasVnrArea.fillna(datatr.MasVnrArea.median(), inplace=True)
datatr.TotalBsmtSF.fillna(datatr.TotalBsmtSF.median(), inplace=True)
datatr.FullBath.fillna(datatr.FullBath.median(), inplace=True)
datatr.TotRmsAbvGrd.fillna(datatr.TotRmsAbvGrd.median(), inplace=True)
datatr.Fireplaces.fillna(datatr.Fireplaces.median(), inplace=True)
datatr.GarageArea.fillna(datatr.GarageArea.median(), inplace=True)


datat.OverallQual.fillna(datat.OverallQual.median(), inplace=True)
datat.YearBuilt.fillna(datat.YearBuilt.median(), inplace=True)
datat.YearRemodAdd.fillna(datat.YearRemodAdd.median(), inplace=True)
datat.MasVnrArea.fillna(datat.MasVnrArea.median(), inplace=True)
datat.TotalBsmtSF.fillna(datat.TotalBsmtSF.median(), inplace=True)
datat.FullBath.fillna(datat.FullBath.median(), inplace=True)
datat.TotRmsAbvGrd.fillna(datat.TotRmsAbvGrd.median(), inplace=True)
datat.Fireplaces.fillna(datat.Fireplaces.median(), inplace=True)
datat.GarageArea.fillna(datat.GarageArea.median(), inplace=True)

boosting = GradientBoostingRegressor(alpha = 0.0001)
boosting.fit(datatr[predictors],datatr['SalePrice'])
y_pred = boosting.predict(datat[predictors])

Id = datat['Id']
SalePrice = y_pred.tolist()

df = pd.DataFrame({'Id': Id, 'SalePrice': SalePrice})
df.to_excel('Submission1.xlsx', sheet_name='sheet1', index=False)
