import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklego.linear_model import LADRegression

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


## Reading the data
train = pd.read_csv('train.csv').drop(columns = ['id'])
test = pd.read_csv('test.csv').drop(columns = ['id'])
sub = pd.read_csv('sample_submission.csv')
original = pd.read_csv('original.csv')

## Appending original data to the training set
train['generated'] = 1
original['generated'] = 0
test['generated'] = 1
train = pd.concat([train, original], axis = 0).reset_index(drop = True)

## Feature Engineering:
train['Male'] = np.where(train['Sex'] == 'M', 1, 0); train['Female'] = np.where(train['Sex'] == 'F', 1, 0)
test['Male'] = np.where(test['Sex'] == 'M', 1, 0); test['Female'] = np.where(test['Sex'] == 'F', 1, 0)

train['Shucked Weight Perc.'] = train['Shucked Weight'] / train['Weight']
train['Viscera Weight Perc.'] = train['Viscera Weight'] / train['Weight']
train['Shell Weight Perc.'] = train['Shell Weight'] / train['Weight']

test['Shucked Weight Perc.'] = test['Shucked Weight'] / test['Weight']
test['Viscera Weight Perc.'] = test['Viscera Weight'] / test['Weight']
test['Shell Weight Perc.'] = test['Shell Weight'] / test['Weight']

numeric_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 
                    'Shell Weight', 'Shucked Weight Perc.', 'Viscera Weight Perc.', 'Shell Weight Perc.']

scaler = StandardScaler().fit(train[numeric_features])
X_train = scaler.transform(train[numeric_features])
X_test = scaler.transform(test[numeric_features])

pca = PCA(0.9).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca = pd.DataFrame(X_train_pca, columns = ['PC_1', 'PC_2', 'PC_3', 'PC_4'])
X_test_pca = pd.DataFrame(X_test_pca, columns = ['PC_1', 'PC_2', 'PC_3', 'PC_4'])

train = pd.concat([train, X_train_pca], axis = 1)
test = pd.concat([test, X_test_pca], axis = 1)

## Dropping some variables
train.drop(columns = ['Sex'], axis = 1, inplace = True)
test.drop(columns = ['Sex'], axis = 1, inplace = True)

## Defining the input and target variables
X = train.drop(columns = ['Age'], axis = 1)
Y = train['Age']

## Defining lists to store results
# gb_cv_scores, gb_preds = list(), list()
hist_cv_scores, hist_preds = list(), list()
lgb_cv_scores, lgb_preds = list(), list()
xgb_cv_scores, xgb_preds = list(), list()
cat_cv_scores, cat_preds = list(), list()
ens_cv_scores, ens_preds = list(), list()

## Performing KFold cross-validation
skf = KFold(n_splits = 10, random_state = 42, shuffle = True)
    
for i, (train_ix, test_ix) in enumerate(skf.split(X, Y)):
        
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
    print('---------------------------------------------------------------')
    
    ######################
    ## GradientBoosting ##
    ######################
        
#     gb_md = GradientBoostingRegressor(loss = 'absolute_error', n_estimators = 800, max_depth = 10, learning_rate = 0.01,
#                                       min_samples_split = 10, min_samples_leaf = 20).fit(X_train, Y_train) 
    
#     gb_pred_1 = gb_md.predict(X_test[X_test['generated'] == 1])
#     gb_pred_2 = gb_md.predict(test)
    
#     gb_score_fold = mean_absolute_error(Y_test[X_test['generated'] == 1], gb_pred_1)
#     gb_cv_scores.append(gb_score_fold)
#     gb_preds.append(gb_pred_2)
    
#     print('Fold', i, '==> GradientBoositng oof MAE is ==>', gb_score_fold)
    
    
    ##########################
    ## HistGradientBoosting ##
    ##########################
        
    hist_md = HistGradientBoostingRegressor(loss = 'absolute_error', l2_regularization = 0.01, early_stopping = True, learning_rate = 0.006,
                                            max_iter = 1000, max_depth = 16, max_bins = 255, min_samples_leaf = 54, n_iter_no_change = 200,
                                            max_leaf_nodes = 57).fit(X_train, Y_train)
    
    hist_pred_1 = hist_md.predict(X_test[X_test['generated'] == 1])
    hist_pred_2 = hist_md.predict(test)

    hist_score_fold = mean_absolute_error(Y_test[X_test['generated'] == 1], hist_pred_1)
    hist_cv_scores.append(hist_score_fold)
    hist_preds.append(hist_pred_2)
    
    print('Fold', i, '==> HistGradient oof MAE is ==>', hist_score_fold)
        
    ##############
    ## LightGBM ##
    ##############
        
    lgb_md = LGBMRegressor(objective = 'mae', n_estimators = 1000, learning_rate = 0.007, num_leaves = 117, reg_alpha = 0.013,
                           reg_lambda = 2.38, subsample = 0.7, colsample_bytree = 0.85, subsample_freq = 4, min_child_samples = 33, 
                           metric = 'mae', boosting_type = 'gbdt').fit(X_train, Y_train)
    
    lgb_pred_1 = lgb_md.predict(X_test[X_test['generated'] == 1])
    lgb_pred_2 = lgb_md.predict(test)

    lgb_score_fold = mean_absolute_error(Y_test[X_test['generated'] == 1], lgb_pred_1)    
    lgb_cv_scores.append(lgb_score_fold)
    lgb_preds.append(lgb_pred_2)
    
    print('Fold', i, '==> LightGBM oof MAE is ==>', lgb_score_fold)
        
    #############
    ## XGBoost ##
    #############
        
    xgb_md = XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 0.9, gamma = 1.6661346939401E-07, learning_rate = 0.005, max_depth = 9, 
                          min_child_weight = 8, n_estimators = 1000, subsample = 0.36, booster = 'gbtree', eta = 0.000078, 
                          grow_policy = 'lossguide', n_jobs = -1, eval_metric = 'mae', verbosity = 0, alpha = 0.0000116).fit(X_train, Y_train)
    
    xgb_pred_1 = xgb_md.predict(X_test[X_test['generated'] == 1])
    xgb_pred_2 = xgb_md.predict(test)

    xgb_score_fold = mean_absolute_error(Y_test[X_test['generated'] == 1], xgb_pred_1)    
    xgb_cv_scores.append(xgb_score_fold)
    xgb_preds.append(xgb_pred_2)
    
    print('Fold', i, '==> XGBoost oof MAE is ==>', xgb_score_fold)
    
    ###############
    ### CatBoost ##
    ###############
    
    cat_md = CatBoostRegressor(iterations = 1000, depth = 7, learning_rate = 0.0045, l2_leaf_reg = 0.11378, random_strength = 0.0179,
                               od_type= 'IncToDec', od_wait = 50, bootstrap_type = 'Bayesian', grow_policy= 'Lossguide', 
                               bagging_temperature = 1.392, eval_metric = 'MAE', loss_function = 'MAE', verbose = False,
                               allow_writing_files = False).fit(X_train, Y_train)
    
    cat_pred_1 = cat_md.predict(X_test[X_test['generated'] == 1])
    cat_pred_2 = cat_md.predict(test)

    cat_score_fold = mean_absolute_error(Y_test[X_test['generated'] == 1], cat_pred_1)    
    cat_cv_scores.append(cat_score_fold)
    cat_preds.append(cat_pred_2)
    
    print('Fold', i, '==> CatBoost oof MAE is ==>', cat_score_fold)
    
    ##################
    ## LAD Ensemble ##
    ##################
    
    x = pd.DataFrame({'hist': hist_pred_1, 'lgb': lgb_pred_1, 'xgb': xgb_pred_1, 'cat': cat_pred_1})
    y = Y_test[X_test['generated'] == 1]
    
    lad_md = LADRegression().fit(x, y)
    lad_pred = lad_md.predict(x)
    
    x_test = pd.DataFrame({'hist': hist_pred_2, 'lgb': lgb_pred_2, 'xgb': xgb_pred_2, 'cat': cat_pred_2})
    lad_pred_test = lad_md.predict(x_test)
        
    ens_score = mean_absolute_error(y, lad_pred)
    ens_cv_scores.append(ens_score)
    ens_preds.append(lad_pred_test)
    
    print('Fold', i, '==> LAD ensemble oof MAE is ==>', ens_score)
    
# print(np.mean(gb_cv_scores))
print(np.mean(hist_cv_scores))
print(np.mean(lgb_cv_scores))
print(np.mean(xgb_cv_scores))
print(np.mean(cat_cv_scores))
print(np.mean(ens_cv_scores))

# gb_preds_test = pd.DataFrame(gb_preds).apply(np.mean, axis = 0)
hist_preds_test = pd.DataFrame(hist_preds).apply(np.mean, axis = 0)
lgb_preds_test = pd.DataFrame(lgb_preds).apply(np.mean, axis = 0)
xgb_preds_test = pd.DataFrame(xgb_preds).apply(np.mean, axis = 0)
cat_preds_test = pd.DataFrame(cat_preds).apply(np.mean, axis = 0)
ens_preds_test = pd.DataFrame(ens_preds).apply(np.mean, axis = 0)

# sub['Age'] = gb_preds_test
# sub.to_csv('submissions/GB.csv', index = False)

sub['Age'] = hist_preds_test
sub.to_csv('submissions/Hist.csv', index = False)

sub['Age'] = lgb_preds_test
sub.to_csv('submissions/LightGBM.csv', index = False)

sub['Age'] = xgb_preds_test
sub.to_csv('submissions/XGBoost.csv', index = False)

sub['Age'] = cat_preds_test
sub.to_csv('submissions/CatBoost.csv', index = False)

sub['Age'] = ens_preds_test
sub.to_csv('submissions/Ensemble.csv', index = False)