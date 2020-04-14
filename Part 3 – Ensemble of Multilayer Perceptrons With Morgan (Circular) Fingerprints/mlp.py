from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np 
import pandas as pd 
import os, path, time



def mlp_model_construction(dict_params_mlp, num_output_nodes,
                           input_length, loss_type, metric):
     
    l1_reg = dict_params_mlp['l1_regularizer']
    l2_reg = dict_params_mlp['l2_regularizer']
     
    input_tensor = Input(shape=(input_length,))
        
    # layer 1
    number_of_nodes = input_length//2
     
    x = Dense(number_of_nodes, activation=dict_params_mlp['activation'],
              kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
              activity_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(input_tensor)
     
    x = Dropout(dict_params_mlp['dropout'])(x)
         
    # layers
    for layer in range(dict_params_mlp['layers'] - 1):
        number_of_nodes = number_of_nodes//2
         
        x = Dense(number_of_nodes, activation=dict_params_mlp['activation'],
                  kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                  activity_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(x)
         
        x = Dropout(dict_params_mlp['dropout'])(x)
 
    output_tensor = Dense(num_output_nodes)(x)
        
    model_mlp_f = Model(input_tensor, output_tensor)
      
    # compile the model
    opt = Adam(lr=0.00025)  # default = 0.001
    model_mlp_f.compile(optimizer=opt, loss=loss_type, metrics=[metric])
      
    return model_mlp_f 


def nn_cv(x_calib, y_calib, cv_folds, metric,
          nn_model, nn_model_copy, batchsize, num_epochs, stop_epochs, verbosity,
          models_dir, model_name, threshold, learn_rate_epochs):
       
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=stop_epochs),                     
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=learn_rate_epochs, 
                      verbose=0, mode='auto', min_lr=1.0e-6)]
     
    cv_num = int(x_calib.shape[0]/cv_folds)
    list_cv_metrics = []
    list_cv_epochs = []
    for fold in range(cv_folds):
        # get train/valid
        x_train = np.vstack((x_calib[:cv_num*fold], x_calib[cv_num*(fold+1):]))        
        y_train = np.hstack((y_calib[:cv_num*fold], y_calib[cv_num*(fold+1):]))
          
        x_valid = x_calib[cv_num*fold:cv_num*(fold+1)]
        y_valid = y_calib[cv_num*fold:cv_num*(fold+1)]
         
        # fit the model
        h = nn_model.fit(x_train, y_train, epochs=num_epochs, 
                         batch_size=batchsize, validation_data=(x_valid, y_valid), 
                         callbacks=callbacks_list, verbose=verbosity)
      
        # collect cv stats
        list_cv_metrics.append(np.min(h.history['val_' + metric]))
        epoch_best_early_stop = len(h.history['val_loss']) - stop_epochs
        list_cv_epochs.append(epoch_best_early_stop)
                  
    dict_return = {}
    mean_cv_error = np.mean(list_cv_metrics)
    max_cv_epochs = np.max(list_cv_epochs)
    print('mean_cv_error =',mean_cv_error,'\tthreshold =',threshold)
    print('max_cv_epochs =',max_cv_epochs)
     
    if mean_cv_error < threshold:
        # fit the model (use copy) with all calib data
        h_refit = nn_model_copy.fit(x_calib, y_calib, epochs=max_cv_epochs, 
                                    batch_size=batchsize, verbose=0)
        nn_model_copy.save(models_dir + model_name + '.h5')
        dict_return['Model Name'] = model_name
        dict_return['Metric'] = metric
        dict_return['Mean CV Error'] = mean_cv_error
        dict_return['Max CV Epochs'] = max_cv_epochs
        dict_return['Batch Size'] = batchsize
      
    return dict_return


if __name__ == '__main__':
     
    calculation_type = 'production'  # production  calibration
    nn_type = 'mlp'
    target_divisor = 10.0  # target E [-0.15, 0.45] so no need of activation function
        
    base_directory = 'YOUR DIRECTORY'
    data_directory = base_directory + 'data/'
     
    results_directory_stub = base_directory + nn_type + '_fingerprints/'
    if not Path(results_directory_stub).is_dir():
        os.mkdir(results_directory_stub)
     
    results_directory = results_directory_stub + 'results_' + nn_type + '_fingerprints/'
    if not Path(results_directory).is_dir():
        os.mkdir(results_directory)
     
    models_directory = results_directory_stub + 'models_' + nn_type + '_fingerprints/'
    if not Path(models_directory).is_dir():
        os.mkdir(models_directory)
 
    metric_type = 'mean_squared_error'
    type_of_loss = metric_type
    verbose = 0  #2
    epochs = 500 #200
    checkpoint_epochs_stop = 15
    learning_rate_epochs = 10
     
    cross_valid_folds = 3
    size_of_parameter_list = 1000
    error_threshold = 0.015
    max_time_minutes = 6*60
    number_of_models = 10
         
    if calculation_type == 'calibration':
        print('\n\n*** starting calibration at',pd.Timestamp.now())
        start_time_calibration = time.time()
             
        # get data
        ycalib = np.load(data_directory + 'y_calib.npy')/target_divisor   
        xcalib = np.load(data_directory + 'x_calib_fingerprints_1d.npy')
        length_of_input = xcalib[0].shape[0] # length_of_fingerprint_vector = 2048
        
        try:
            number_of_outputs = ycalib.shape[1]
        except:
            number_of_outputs = 1  # ycalib.shape[1]
         
        # set up parameters (stochastic)
        list_param_dict = parameters_mlp(size_of_parameter_list)
        list_dict_results = []
        counter = 0
        # loop over parameters
        for dictionary_parameter in list_param_dict:   
             
            modelname = nn_type + '_fingerprints_' + str(counter)
            counter = counter + 1
                 
            print('\n\n*** starting at',pd.Timestamp.now())
            start_time_loop = time.time()            
             
            model_nn = mlp_model_construction(dictionary_parameter, 
                                                 number_of_outputs, length_of_input,
                                                 type_of_loss, metric_type) 
            print(dictionary_parameter)                     
             
            dict_results = nn_cv(xcalib, ycalib, 
                                    cross_valid_folds, metric_type,
                                    model_nn, model_nn, 
                                    dictionary_parameter['batch_size'], epochs, 
                                    checkpoint_epochs_stop, verbose,
                                    models_directory, modelname, error_threshold,
                                    learning_rate_epochs)
             
            elapsed_time_minutes = (time.time() - start_time_loop)/60
            print('*** elapsed time =',elapsed_time_minutes,' mins\tcounter =',counter-1)
             
            # add epochs and modelname to param dict and save it
            if len(dict_results) > 0:
                dictionary_parameter.update(dict_results)
                list_dict_results.append(dictionary_parameter)
                 
            # check elapsed time for early stopping
            elapsed_time_minutes = (time.time() - start_time_calibration)/60
            if elapsed_time_minutes > max_time_minutes:
                break
         
        print('\n\n*** ending calibation at',pd.Timestamp.now())
        print('elapsed time =',(time.time()-start_time_calibration)/60,' min')
         
        # collect results in a df, then save
        df_param_results = pd.DataFrame(data=list_dict_results)
        df_param_results.to_pickle(models_directory + 'df_parameter_results.pkl')
        df_param_results.to_csv(models_directory + 'df_parameter_results.csv')
        list_dict_results.clear()
             
    elif calculation_type == 'production':
         
        # get data
        yprod = np.load(data_directory + 'y_prod.npy')  # UNALTERED (SEE CREATE_ENSEMBLE)
        xprod = np.load(data_directory + 'x_prod_fingerprints_1d.npy')
        length_of_input = xprod[0].shape[0] # length_of_fingerprint_vector = 2048
             
        df_results = pd.read_pickle(models_directory + 'df_parameter_results.pkl')
                       
        neural_network_ensemble.create_ensemble('regression', 
                       'Mean CV Error', 'Keras', 
                       df_results, number_of_models, models_directory, 
                       results_directory, xprod, yprod, target_divisor)
         
    else:
        raise NameError