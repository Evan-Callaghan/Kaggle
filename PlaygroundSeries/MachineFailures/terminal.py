## ENZYMES ##

## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)

from flaml import AutoML
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

## Reading the data
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sub = pd.read_csv('Data/sample_submission.csv')

## Defining input and target variables
inputs = train.drop(columns = ['id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']).columns
targets = ['EC1', 'EC2']


## EC1:

## Defining input and target variables for training
X = train[inputs]
Y = train['EC1']
X_test = test[inputs]

## Creating lists to store results
aml_lgb_cv_scores, aml_lgb_preds = list(), list()
aml_cat_cv_scores, aml_cat_preds = list(), list()
ens_cv_scores, ens_preds = list(), list()

## Performing stratified k fold
skf = StratifiedKFold(n_splits = 15, random_state = 42, shuffle = True)
    
for i, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
    
    print('---------------------------------------------------------------')
    
    ## FLAML (LGBM) ##
    
    automl = AutoML()
    automl_settings = {'time_budget': 120,  
                       'metric': 'roc_auc',
                       'task': 'classification',
                       'estimator_list': ['lgbm'],
                       "log_file_name": '',
                      }

    automl.fit(X_train = X_train, y_train = Y_train, **automl_settings, verbose = False)
    
    aml_lgb_pred_1 = automl.predict_proba(X_valid)[:, 1]
    aml_lgb_pred_2 = automl.predict_proba(X_test)[:, 1]

    aml_lgb_score_fold = roc_auc_score(Y_valid, aml_lgb_pred_1)
    aml_lgb_cv_scores.append(aml_lgb_score_fold)
    aml_lgb_preds.append(aml_lgb_pred_2)
    
    print('Fold', i+1, '==> FLAML (LGBM) oof ROC-AUC is ==>', aml_lgb_score_fold)
    
    ## FLAML (CatBoost) ##
    
    automl = AutoML()
    automl_settings = {'time_budget': 120,  
                       'metric': 'roc_auc',
                       'task': 'classification',
                       'estimator_list': ['catboost'],
                       "log_file_name": '',
                      }

    automl.fit(X_train = X_train, y_train = Y_train, **automl_settings, verbose = False)
    
    aml_cat_pred_1 = automl.predict_proba(X_valid)[:, 1]
    aml_cat_pred_2 = automl.predict_proba(X_test)[:, 1]
    
    aml_cat_score_fold = roc_auc_score(Y_valid, aml_cat_pred_1)
    aml_cat_cv_scores.append(aml_cat_score_fold)
    aml_cat_preds.append(aml_cat_pred_2)
    
    print('Fold', i+1, '==> FLAML (CatBoost) oof ROC-AUC is ==>', aml_cat_score_fold)
    
    ######################
    ## Average Ensemble ##
    ######################
    
    ens_pred_1 = (aml_lgb_pred_1 + 2 * aml_cat_pred_1) / 3
    ens_pred_2 = (aml_lgb_pred_2 + 2 * aml_cat_pred_2) / 3
    
    ens_score_fold = roc_auc_score(Y_valid, ens_pred_1)
    ens_cv_scores.append(ens_score_fold)
    ens_preds.append(ens_pred_2)
    
    print('Fold', i+1, '==> Average Ensemble oof ROC-AUC score is ==>', ens_score_fold)
    
flaml_lgb = np.mean(aml_lgb_cv_scores)
flaml_cat = np.mean(aml_cat_cv_scores)
ens_cv_score = np.mean(ens_cv_scores)

print('LGBM: ', flaml_lgb)
print('CAT: ', flaml_cat)
print('ENSEMBLE: ', ens_cv_score)

lgb_preds_EC1 = pd.DataFrame(aml_lgb_preds).apply(np.mean, axis = 0)
cat_preds_EC1 = pd.DataFrame(aml_cat_preds).apply(np.mean, axis = 0)
ens_preds_EC1 = pd.DataFrame(ens_preds).apply(np.mean, axis = 0)


## EC2:

## Defining input and target variables for training
X = train[inputs]
Y = train['EC2']
X_test = test[inputs]

## Creating lists to store results
aml_lgb_cv_scores, aml_lgb_preds = list(), list()
aml_cat_cv_scores, aml_cat_preds = list(), list()
ens_cv_scores, ens_preds = list(), list()

## Performing stratified k fold
skf = StratifiedKFold(n_splits = 15, random_state = 42, shuffle = True)
    
for i, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
    
    print('---------------------------------------------------------------')
    
    ## FLAML (LGBM) ##
    
    automl = AutoML()
    automl_settings = {'time_budget': 120,  
                       'metric': 'roc_auc',
                       'task': 'classification',
                       'estimator_list': ['lgbm'],
                       "log_file_name": '',
                      }

    automl.fit(X_train = X_train, y_train = Y_train, **automl_settings, verbose = False)
    
    aml_lgb_pred_1 = automl.predict_proba(X_valid)[:, 1]
    aml_lgb_pred_2 = automl.predict_proba(X_test)[:, 1]

    aml_lgb_score_fold = roc_auc_score(Y_valid, aml_lgb_pred_1)
    aml_lgb_cv_scores.append(aml_lgb_score_fold)
    aml_lgb_preds.append(aml_lgb_pred_2)
    
    print('Fold', i+1, '==> FLAML (LGBM) oof ROC-AUC is ==>', aml_lgb_score_fold)
    
    ## FLAML (CatBoost) ##
    
    automl = AutoML()
    automl_settings = {'time_budget': 120,  
                       'metric': 'roc_auc',
                       'task': 'classification',
                       'estimator_list': ['catboost'],
                       "log_file_name": '',
                      }

    automl.fit(X_train = X_train, y_train = Y_train, **automl_settings, verbose = False)
    
    aml_cat_pred_1 = automl.predict_proba(X_valid)[:, 1]
    aml_cat_pred_2 = automl.predict_proba(X_test)[:, 1]
    
    aml_cat_score_fold = roc_auc_score(Y_valid, aml_cat_pred_1)
    aml_cat_cv_scores.append(aml_cat_score_fold)
    aml_cat_preds.append(aml_cat_pred_2)
    
    print('Fold', i+1, '==> FLAML (CatBoost) oof ROC-AUC is ==>', aml_cat_score_fold)
    
    ######################
    ## Average Ensemble ##
    ######################
    
    ens_pred_1 = (aml_lgb_pred_1 + 2 * aml_cat_pred_1) / 3
    ens_pred_2 = (aml_lgb_pred_2 + 2 * aml_cat_pred_2) / 3
    
    ens_score_fold = roc_auc_score(Y_valid, ens_pred_1)
    ens_cv_scores.append(ens_score_fold)
    ens_preds.append(ens_pred_2)
    
    print('Fold', i+1, '==> Average Ensemble oof ROC-AUC score is ==>', ens_score_fold)
    
flaml_lgb = np.mean(aml_lgb_cv_scores)
flaml_cat = np.mean(aml_cat_cv_scores)
ens_cv_score = np.mean(ens_cv_scores)

print('LGBM: ', flaml_lgb)
print('CAT: ', flaml_cat)
print('ENSEMBLE: ', ens_cv_score)

lgb_preds_EC2 = pd.DataFrame(aml_lgb_preds).apply(np.mean, axis = 0)
cat_preds_EC2 = pd.DataFrame(aml_cat_preds).apply(np.mean, axis = 0)
ens_preds_EC2 = pd.DataFrame(ens_preds).apply(np.mean, axis = 0)


## Saving prediction files:
sub['EC1'] = lgb_preds_EC1
sub['EC2'] = lgb_preds_EC2
sub.to_csv('Submissions/LGBM_baseline.csv', index = False)

sub['EC1'] = lgb_preds_EC1 
sub['EC2'] = cat_preds_EC2
sub.to_csv('Submissions/LGBM_CAT_baseline.csv', index = False)

sub['EC1'] = lgb_preds_EC1 
sub['EC2'] = ens_preds_EC2
sub.to_csv('Submissions/LGBM_ENS_baseline.csv', index = False)

sub['EC1'] = cat_preds_EC1 
sub['EC2'] = lgb_preds_EC2
sub.to_csv('Submissions/CAT_LGBM_baseline.csv', index = False)

sub['EC1'] = cat_preds_EC1 
sub['EC2'] = cat_preds_EC2
sub.to_csv('Submissions/CAT_baseline.csv', index = False)

sub['EC1'] = cat_preds_EC1 
sub['EC2'] = ens_preds_EC2
sub.to_csv('Submissions/CAT_ENS_baseline.csv', index = False)

sub['EC1'] = ens_preds_EC1 
sub['EC2'] = lgb_preds_EC2
sub.to_csv('Submissions/ENS_LGBM_baseline.csv', index = False)

sub['EC1'] = ens_preds_EC1 
sub['EC2'] = cat_preds_EC2
sub.to_csv('Submissions/ENS_CAT_baseline.csv', index = False)

sub['EC1'] = ens_preds_EC1 
sub['EC2'] = ens_preds_EC2
sub.to_csv('Submissions/ENS_baseline.csv', index = False)