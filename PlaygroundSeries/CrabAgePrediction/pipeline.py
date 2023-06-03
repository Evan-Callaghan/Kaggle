import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split 
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklego.linear_model import LADRegression
from sklearn.decomposition import PCA

train = pd.read_csv('train.csv').drop(columns = ['id'])
test = pd.read_csv('test.csv').drop(columns = ['id'])
sub = pd.read_csv('sample_submission.csv')
original = pd.read_csv('original.csv')

train = pd.concat([train, original], axis = 0).reset_index(drop = True)

## Weight
train['Accounted Weight'] = train['Shucked Weight'] + train['Viscera Weight'] + train['Shell Weight']
train['Weight Diff.'] = train['Weight'] - train['Accounted Weight']
train['Too Heavy'] = np.where(train['Accounted Weight'] > train['Weight'], 1, 0).astype(int)
train['Shucked Weight'] = np.where(train['Accounted Weight'] > train['Weight'], 0.424150 * train['Weight'], train['Shucked Weight'])
train['Viscera Weight'] = np.where(train['Accounted Weight'] > train['Weight'], 0.213569 * train['Weight'], train['Viscera Weight'])
train['Shell Weight'] = np.where(train['Accounted Weight'] > train['Weight'], 0.288712 * train['Weight'], train['Shell Weight'])
train['Shucked Weight Perc.'] = train['Shucked Weight'] / train['Weight']
train['Viscera Weight Perc.'] = train['Viscera Weight'] / train['Weight']
train['Shell Weight Perc.'] = train['Shell Weight'] / train['Weight']

test['Accounted Weight'] = test['Shucked Weight'] + test['Viscera Weight'] + test['Shell Weight']
test['Weight Diff.'] = test['Weight'] - test['Accounted Weight']
test['Too Heavy'] = np.where(test['Accounted Weight'] > test['Weight'], 1, 0).astype(int)
test['Shucked Weight'] = np.where(test['Accounted Weight'] > test['Weight'], 0.424150 * test['Weight'], test['Shucked Weight'])
test['Viscera Weight'] = np.where(test['Accounted Weight'] > test['Weight'], 0.213569 * test['Weight'], test['Viscera Weight'])
test['Shell Weight'] = np.where(test['Accounted Weight'] > test['Weight'], 0.288712 * test['Weight'], test['Shell Weight'])
test['Shucked Weight Perc.'] = test['Shucked Weight'] / test['Weight']
test['Viscera Weight Perc.'] = test['Viscera Weight'] / test['Weight']
test['Shell Weight Perc.'] = test['Shell Weight'] / test['Weight']

## Dimensions
train['Height'] = np.where(train['Height'] > 2, np.mean(train['Height']), 
                           np.where(train['Height'] == 0, 0.29337*train['Length']-0.03826729, train['Height']))
train['Volume'] = train['Length'] * train['Diameter'] * train['Height']
train['Density'] = train['Weight'] / train['Volume']

test['Height'] = np.where(test['Height'] > 2, np.mean(test['Height']), 
                           np.where(test['Height'] == 0, 0.29400666*test['Length']-0.03933592, test['Height']))
test['Volume'] = test['Length'] * test['Diameter'] * test['Height']
test['Density'] = test['Weight'] / test['Volume']

## Gender
train['Male'] = np.where(train['Sex'] == 'M', 1, 0); train['Female'] = np.where(train['Sex'] == 'F', 1, 0)
test['Male'] = np.where(test['Sex'] == 'M', 1, 0); test['Female'] = np.where(test['Sex'] == 'F', 1, 0)

## PCA
numeric_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Accounted Weight', 
                    'Weight Diff.', 'Shucked Weight Perc.', 'Viscera Weight Perc.', 'Shell Weight Perc.', 'Volume', 'Density']

scaler = StandardScaler().fit(train[numeric_features])
X_train = scaler.transform(train[numeric_features])
X_test = scaler.transform(test[numeric_features])

pca = PCA(4).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca = pd.DataFrame(X_train_pca, columns = ['PC_1', 'PC_2', 'PC_3', 'PC_4'])
X_test_pca = pd.DataFrame(X_test_pca, columns = ['PC_1', 'PC_2', 'PC_3', 'PC_4'])

train = pd.concat([train, X_train_pca], axis = 1)
test = pd.concat([test, X_test_pca], axis = 1)

train = train.drop(columns = ['Sex', 'Too Heavy'])
test = test.drop(columns = ['Sex', 'Too Heavy'])


## Defining input and target variables
X = train.drop(columns = ['Age'])
Y = train['Age']

## Initializing parameters
SEED = 42
SPLITS = 10

## Defining Optuna objective functions
def HIST_objective(trial):

    ## Defining the hyper-parameter grid
    param_grid = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01), 
                  'max_iter': trial.suggest_int('max_iter', 100, 1000, 50), 
                  'max_depth': trial.suggest_int('max_depth', 3, 12),  
                  'l2_regularization': trial.suggest_float('l2_regularization', 0, 0.1, step = 0.002), 
                  'random_state': trial.suggest_int('random_state', 1, 500),
                 }
    scores = list()
    kf = KFold(n_splits = SPLITS, shuffle = True, random_state = SEED)
    
    for train_idx, valid_idx in kf.split(X, Y):
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
        
        ## Building the model
        model = HistGradientBoostingRegressor(**param_grid, loss = 'absolute_error', early_stopping = True).fit(X_train, Y_train)
        
        ## Predicting on the test data-frame
        preds = model.predict(X_valid)
        
        ## Evaluating model performance on the test set
        scores.append(mean_absolute_error(Y_valid, preds))
    
    return np.mean(scores)

def XGB_objective(trial):

    ## Defining the hyper-parameter grid
    param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50),  
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),  
                  'max_depth': trial.suggest_int('max_depth', 3, 12),  
                  'gamma': trial.suggest_float('gamma', 0, 0.3, step = 0.05),  
                  'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  
                  'subsample': trial.suggest_float('subsample', 0.6, 1, step = 0.05),  
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05), 
                  'seed': trial.suggest_int('seed', 1, 1000) 
                 }
    scores = list()
    kf = KFold(n_splits = SPLITS, shuffle = True, random_state = SEED)
    
    for train_idx, valid_idx in kf.split(X, Y):
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
        
        ## Building the model
        model = XGBRegressor(**param_grid, n_jobs = -1).fit(X_train, Y_train)
        
        ## Predicting on the test data-frame
        preds = model.predict(X_valid)
        
        ## Evaluating model performance on the test set
        scores.append(mean_absolute_error(Y_valid, preds))
    
    return np.mean(scores)

def LGBM_objective(trial):
    
    ## Defining the hyper-parameter grid
    param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50), 
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01), 
                  'num_leaves': trial.suggest_int('num_leaves', 5, 40, step = 1), 
                  'max_depth': trial.suggest_int('max_depth', 3, 12), 
                  'subsample': trial.suggest_float('subsample', 0.6, 1, step = 0.05),  
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05), 
                  'random_state': trial.suggest_int('random_state', 1, 1000),
                 }
    scores = list()
    kf = KFold(n_splits = SPLITS, shuffle = True, random_state = SEED)
    
    for train_idx, valid_idx in kf.split(X, Y):
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
        
        ## Building the model
        model = LGBMRegressor(**param_grid, n_jobs = -1, verbosity = -1).fit(X_train, Y_train)
        
        ## Predicting on the test data-frame
        preds = model.predict(X_valid)
        
        ## Evaluating model performance on the test set
        scores.append(mean_absolute_error(Y_valid, preds))
    
    return np.mean(scores)


## Starting HistGradientBoosting
## ----
## Creating a study object and to optimize the home objective function
study_hist = optuna.create_study(direction = 'minimize', study_name = 'HistGradientBoosting')
study_hist.optimize(HIST_objective, n_trials = 500)

## Starting XGBoost
## ----
## Creating a study object and to optimize the home objective function
study_xgb = optuna.create_study(direction = 'minimize', study_name = 'XGBoost')
study_xgb.optimize(XGB_objective, n_trials = 50)

## Starting LightGBM
## ----
## Creating a study object and to optimize the home objective function
study_lgbm = optuna.create_study(direction = 'minimize', study_name = 'LightGBM')
study_lgbm.optimize(LGBM_objective, n_trials = 500)

## Printing best hyper-parameter set
print('HistGB: \n', study_hist.best_trial.params)
print(study_hist.best_trial.value)

## Printing best hyper-parameter set
print('\nXGBoost: \n', study_xgb.best_trial.params)
print(study_xgb.best_trial.value)

## Printing best hyper-parameter set
print('\nLightGBM: \n', study_lgbm.best_trial.params)
print(study_lgbm.best_trial.value)


## Storing optimal HP sets
hist_params = study_hist.best_trial.params
xgb_params = study_xgb.best_trial.params
lgbm_params = study_lgbm.best_trial.params

## Defining the input and target variables
X = train.drop(columns = ['Age'], axis = 1)
Y = train['Age']

## Defining lists to store results
hist_cv_scores, hist_preds = list(), list()
hist_cv_scores_round, hist_preds_round = list(), list()

lgb_cv_scores, lgb_preds = list(), list()
lgb_cv_scores_round, lgb_preds_round = list(), list()

xgb_cv_scores, xgb_preds = list(), list()
xgb_cv_scores_round, xgb_preds_round = list(), list()

ens_cv_scores, ens_preds = list(), list()
ens_cv_scores_round, ens_preds_round = list(), list()

ens_cv_scores2, ens_preds2 = list(), list()
ens_cv_scores_round2, ens_preds_round2 = list(), list()


## Performing KFold cross-validation
kf = KFold(n_splits = SPLITS, shuffle = True)
    
for i, (train_ix, test_ix) in enumerate(kf.split(X, Y)):
        
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    
    print('---------------------------------------------------------------')    
    
    hist_md = HistGradientBoostingRegressor(**hist_params, loss = 'absolute_error', early_stopping = True).fit(X_train, Y_train)
    
    hist_pred_1 = hist_md.predict(X_test); hist_pred_1_round = np.round_(hist_pred_1).astype(int)
    hist_pred_2 = hist_md.predict(test); hist_pred_2_round = np.round_(hist_pred_2).astype(int)
    
    hist_score_fold = mean_absolute_error(Y_test, hist_pred_1); hist_score_fold_round = mean_absolute_error(Y_test, hist_pred_1_round)
    hist_cv_scores.append(hist_score_fold); hist_cv_scores_round.append(hist_score_fold_round)
    hist_preds.append(hist_pred_2); hist_preds_round.append(hist_pred_2_round)
    
    print('Fold', i+1, '==> HistGradient oof MAE is ==>', hist_score_fold)
    print('Fold', i+1, '==> HistGradient oof MAE is ==>', hist_score_fold_round)
    
    
    lgb_md = LGBMRegressor(**lgbm_params, n_jobs = -1, verbosity = -1).fit(X_train, Y_train)

    lgb_pred_1 = lgb_md.predict(X_test); lgb_pred_1_round = np.round_(lgb_pred_1).astype(int)
    lgb_pred_2 = lgb_md.predict(test); lgb_pred_2_round = np.round_(lgb_pred_2).astype(int)
    
    lgb_score_fold = mean_absolute_error(Y_test, lgb_pred_1); lgb_score_fold_round = mean_absolute_error(Y_test, lgb_pred_1_round)
    lgb_cv_scores.append(lgb_score_fold); lgb_cv_scores_round.append(lgb_score_fold_round)
    lgb_preds.append(lgb_pred_2); lgb_preds_round.append(lgb_pred_2_round)
    
    print('Fold', i+1, '==> LightGBM oof MAE is ==>', lgb_score_fold)
    print('Fold', i+1, '==> LightGBM oof MAE is ==>', lgb_score_fold_round)
    
    
    xgb_md = XGBRegressor(**xgb_params, n_jobs = -1).fit(X_train, Y_train)
    
    xgb_pred_1 = xgb_md.predict(X_test); xgb_pred_1_round = np.round_(xgb_pred_1).astype(int)
    xgb_pred_2 = xgb_md.predict(test); xgb_pred_2_round = np.round_(xgb_pred_2).astype(int)
    
    xgb_score_fold = mean_absolute_error(Y_test, xgb_pred_1); xgb_score_fold_round = mean_absolute_error(Y_test, xgb_pred_1_round)
    xgb_cv_scores.append(xgb_score_fold); xgb_cv_scores_round.append(xgb_score_fold_round)
    xgb_preds.append(xgb_pred_2); xgb_preds_round.append(xgb_pred_2_round)
    
    print('Fold', i+1, '==> XGBoost oof MAE is ==>', xgb_score_fold)
    print('Fold', i+1, '==> XGBoost oof MAE is ==>', xgb_score_fold_round)
    
    
    x = pd.DataFrame({'HIST': hist_pred_1, 'LGB': lgb_pred_1, 'XGB': xgb_pred_1})
    y = Y_test
    
    lad_md = LADRegression().fit(x, y)
    lad_pred = lad_md.predict(x); lad_pred_round = np.round_(lad_pred).astype(int)
    
    x_test = pd.DataFrame({'HIST': hist_pred_2, 'LGB': lgb_pred_2, 'XGB': xgb_pred_2})
    lad_pred_test = lad_md.predict(x_test); lad_pred_test_round = np.round_(lad_pred_test).astype(int)
        
    ens_score = mean_absolute_error(y, lad_pred); ens_score_round = mean_absolute_error(y, lad_pred_round)
    ens_cv_scores.append(ens_score); ens_cv_scores_round.append(ens_score_round)
    ens_preds.append(lad_pred_test); ens_preds_round.append(lad_pred_test_round)
    
    print('Fold', i+1, '==> LAD ensemble oof MAE is ==>', ens_score)
    print('Fold', i+1, '==> LAD ensemble oof MAE is ==>', ens_score_round)
    
    
    x = pd.DataFrame({'HIST': hist_pred_1_round, 'LGB': lgb_pred_1_round, 'XGB': xgb_pred_1_round})
    y = Y_test
    
    lad_md = LADRegression().fit(x, y)
    lad_pred = lad_md.predict(x); lad_pred_round = np.round_(lad_pred).astype(int)
    
    x_test = pd.DataFrame({'HIST': hist_pred_2_round, 'LGB': lgb_pred_2_round, 'XGB': xgb_pred_2_round})
    lad_pred_test = lad_md.predict(x_test); lad_pred_test_round = np.round_(lad_pred_test).astype(int)
    
    ens_score = mean_absolute_error(y, lad_pred); ens_score_round = mean_absolute_error(y, lad_pred_round)
    ens_cv_scores2.append(ens_score); ens_cv_scores_round2.append(ens_score_round)
    ens_preds2.append(lad_pred_test); ens_preds_round2.append(lad_pred_test_round)
    
    print('Fold', i+1, '==> LAD Rounded ensemble oof MAE is ==>', ens_score)
    print('Fold', i+1, '==> LAD Rounded ensemble oof MAE is ==>', ens_score_round)

print(np.mean(hist_cv_scores), ' --> ', np.mean(hist_cv_scores_round))
print(np.mean(lgb_cv_scores), ' --> ', np.mean(lgb_cv_scores_round))
print(np.mean(xgb_cv_scores), ' --> ', np.mean(xgb_cv_scores_round))
print(np.mean(ens_cv_scores), ' --> ', np.mean(ens_cv_scores_round))
print(np.mean(ens_cv_scores2), ' --> ', np.mean(ens_cv_scores_round2))

hist = pd.DataFrame(hist_preds).apply(np.mean, axis = 0); hist_round = pd.DataFrame(hist_preds_round).apply(np.mean, axis = 0)
lgb = pd.DataFrame(lgb_preds).apply(np.mean, axis = 0); lgb_round = pd.DataFrame(lgb_preds_round).apply(np.mean, axis = 0)
xgb = pd.DataFrame(xgb_preds).apply(np.mean, axis = 0); xgb_round = pd.DataFrame(xgb_preds_round).apply(np.mean, axis = 0)
ens = pd.DataFrame(ens_preds).apply(np.mean, axis = 0); ens_round = pd.DataFrame(ens_preds_round).apply(np.mean, axis = 0)
ens2 = pd.DataFrame(ens_preds2).apply(np.mean, axis = 0); ens_round2 = pd.DataFrame(ens_preds_round2).apply(np.mean, axis = 0)

sub['Age'] = hist
sub.to_csv('submissions/Hist_sub.csv', index = False)

sub['Age'] = np.round(hist_round)
sub.to_csv('submissions/Hist_sub_round.csv', index = False)

sub['Age'] = lgb
sub.to_csv('submissions/LGBM_sub.csv', index = False)

sub['Age'] = np.round(lgb_round)
sub.to_csv('submissions/LGBM_sub_round.csv', index = False)

sub['Age'] = xgb
sub.to_csv('submissions/XGB_sub.csv', index = False)

sub['Age'] = np.round(xgb_round)
sub.to_csv('submissions/XGB_sub_round.csv', index = False)

sub['Age'] = ens
sub.to_csv('submissions/Ens_sub.csv', index = False)

sub['Age'] = np.round(ens_round)
sub.to_csv('submissions/Ens_sub_round.csv', index = False)

sub['Age'] = ens2
sub.to_csv('submissions/Ens_sub2.csv', index = False)

sub['Age'] = np.round(ens_round2)
sub.to_csv('submissions/Ens_sub_round2.csv', index = False)