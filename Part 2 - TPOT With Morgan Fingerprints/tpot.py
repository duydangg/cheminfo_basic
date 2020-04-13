import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
 
from sklearn.metrics import mean_squared_error  # ‘neg_mean_squared_error’
from sklearn.metrics import median_absolute_error  # ‘neg_median_absolute_error’
from sklearn.metrics import r2_score
 
from tpot import TPOTRegressor
     
# *******************************************************************
 
def tpot_dictionary(num_input, step):
     
    tpot_config_dict = {
 
    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
 
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_features': np.arange(0.05, 1.01, 0.1),
        'min_samples_split': range(2, 13),
        'min_samples_leaf': range(1, 13),
        'bootstrap': [True, False]
    },
 
    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [50, 100, 150, 200, 250],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 13),
        'min_samples_leaf': range(1, 13),
        'subsample': np.arange(0.05, 1.01, 0.1),
        'max_features': np.arange(0.05, 1.01, 0.1),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },
 
    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [50, 100, 150, 200, 250],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },
 
    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': np.arange(1,101,2,dtype='int'),
        'weights': ["uniform", "distance"],
        'p': [2]
    },
 
    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },
 
    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },
 
    'sklearn.svm.SVR': {
        'kernel': ['rbf'],
        'cache_size': [1000],
        'shrinking': [True, False],
        'gamma': [5.0e-5, 1.0e-2, 1.0, 1.0e3],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1.0e-3, 1.0e-1, 1.0, 100.0, 1.0e4],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },
 
    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_features': np.arange(0.05, 1.01, 0.2),
        'min_samples_split': range(2, 13),
        'min_samples_leaf': range(1, 13),
        'bootstrap': [True, False]
    },
 
    'sklearn.linear_model.RidgeCV': {
    },
 
    'xgboost.XGBRegressor': {
        'n_estimators': np.linspace(25,250,num=10,endpoint=True,dtype='int'),
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.1),
        'min_child_weight': range(1, 13),
        'colsample_bylevel': [0.7, 0.85, 1.0],
        'nthread': [1],
        'objective': ['reg:squarederror']
    },
 
    # Preprocesssors
 
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },
 
#    'sklearn.preprocessing.MaxAbsScaler': {
#    },
#
#    'sklearn.preprocessing.MinMaxScaler': {
#    },
#
#    'sklearn.preprocessing.Normalizer': {
#        'norm': ['l1', 'l2', 'max']
#    },
 
    'sklearn.decomposition.PCA': {
        'n_components': np.arange(num_input//2, num_input+1, 
                                  step, dtype='int'),
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
 
#    'sklearn.preprocessing.PolynomialFeatures': {
#        'degree': [2,3],
#        'include_bias': [False],
#        'interaction_only': [False]
#    },
#
#    'sklearn.kernel_approximation.RBFSampler': {
#        'gamma': np.arange(0.0, 1.01, 0.05)
#    },
#
#    'sklearn.preprocessing.RobustScaler': {
#    },
#
#    'sklearn.preprocessing.StandardScaler': {
#    },
 
 
    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },
 
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': np.arange(5,101,5,dtype=int),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },
 
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
 
    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.1),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.1, 1.01, 0.1)
            }
        }
    }
 
    }
     
         
    return tpot_config_dict
     
# end of tpot_dictionary()    
 
# *******************************************************************
 
def tpot_regression(x_calib, y_calib, x_prod, y_prod, results_direct,
                    cv_folds, error_metric, num_jobs, gens, pop,
                    mins, mins_per_pipeline, verbose, early_stop_generations,
                    tpot_config_dict, model_name='tpot_best'):
         
    checkpoint_folder = results_direct + 'checkpoint_folder/'
    if not Path(checkpoint_folder).is_dir():
        os.mkdir(checkpoint_folder)
        
    ml_model = TPOTRegressor(generations=gens, population_size=pop, scoring=error_metric,
                             max_time_mins=mins, cv=cv_folds, verbosity=verbose,
                             n_jobs=num_jobs, early_stop=early_stop_generations,
                             max_eval_time_mins=mins_per_pipeline,
                             config_dict=tpot_config_dict,
                             periodic_checkpoint_folder=checkpoint_folder)
     
    ml_model.fit(x_calib, y_calib)
     
    # save entire pipeline
    ml_model.export(results_direct + model_name + '.py')
    joblib.dump(ml_model.fitted_pipeline_, results_direct + model_name + '.sav')
    
    # for cross valdation errors see the exported model py file
         
    # production - results and errors
    y_prod_predict = ml_model.predict(x_prod)
    np.save(results_direct + model_name + '_prod_predicted.npy', y_prod_predict)
     
    df_prod_errors = pd.DataFrame(index=['Mean Squared Error','Median Absolute Error',
                                         'Correlation Coefficient','R2'])
    df_prod_errors['TPOT Best'] = [mean_squared_error(y_prod, y_prod_predict),
                                   median_absolute_error(y_prod, y_prod_predict),
                                   np.corrcoef(y_prod, y_prod_predict)[0][-1],
                                   r2_score(y_prod, y_prod_predict)]
    df_prod_errors.to_csv(results_direct + model_name + '_prod_errors.csv')        
         
# end of tpot_regression()
 
# *******************************************************************
         
if __name__ == '__main__':
         
    calibration_fraction = 0.8
    base_directory = '/home/dangnd/project/cheminfo_basic/Part 2 - TPOT With Morgan Fingerprints'
    data_directory = base_directory + 'lipo_tpot_aws/'
     
    results_directory = base_directory + 'results_tpot_calibration/'
    if not Path(results_directory).is_dir():
        os.mkdir(results_directory)
     
    # get data
    xcalib = np.load(data_directory + 'x_calib_fingerprints_1d.npy')
    ycalib = np.load(data_directory + 'y_calib.npy')
     
    xprod = np.load(data_directory + 'x_prod_fingerprints_1d.npy')
    yprod = np.load(data_directory + 'y_prod.npy')
         
    # tpot parameters                
    cvfolds = 3
    errormetric = 'neg_mean_squared_error'
    numjobs = -1
    generations = 1000
    population_size = 100
    max_time_mins = 8*60
    max_mins_per_pipeline = 10
    early_stop_gens = 10
    verbosity = 1
         
    print('\nstart time:',pd.Timestamp.now())
     
    num_features = xcalib.shape[1]
    step_size = 32
     
    tpot_configuration_dictionary = tpot_dictionary(num_features, step_size)
 
    tpot_regression(xcalib, ycalib, xprod, yprod, results_directory,
                    cvfolds, errormetric, numjobs, generations, population_size,
                    max_time_mins, max_mins_per_pipeline, verbosity, early_stop_gens,
                    tpot_configuration_dictionary)
 
    print('\nending at:',pd.Timestamp.now())