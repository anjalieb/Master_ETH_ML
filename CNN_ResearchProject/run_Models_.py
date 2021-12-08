###########################################################################################
# Libraries
###########################################################################################
import pandas as pd
import os
from tensorflow import keras
from keras.utils import np_utils
import numpy as np
from keras.models import Model
import tensorflow as tf
import argparse

from sklearn.utils import class_weight
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
from pathlib import Path
from numpy.random import seed
import torch
from models import CNN_Model
from utilities import evalBinaryClassifierCNN, plot_trainingProcess
from dataloader import DataGenerator




########################################################################################################################
# Global parameters
########################################################################################################################

wd   = os.getcwd()
# set path to data
path = r'E:\labrot_shared\codeAnja'

# define number of patches per ID to be included
num_patches = 1000


########################################################################################################################
# load csv file containing metadata
Y        = pd.read_csv(r'E:\labrot_shared\data\TCIA_LGG_cases_115.csv')
lengths = Y[['Filename','Length']]

# function that selects randomly the num_patches, return a dict containing the ID and the patch number
def load_patches(partition_spec, num_patche, runIDuse):
    # random seed set on run Index
    seed(runIDuse)
    dict_pat_all = {}
    dict_pat_all = list(dict_pat_all)

    for every in range(len(partition_spec)):
        
        ind = np.where(lengths['Filename'] == partition_spec[every])
        res = lengths.loc[ind[0][0]]
        limit=res[1]
        if limit < num_patches:
            # those patients having less patches than num_patches
            rand_patches = np.arange(limit)
            listID =[partition_spec[every] for i in range(limit)]
        else:
            rand_patches = np.random.choice(limit,num_patches)
            listID =[partition_spec[every] for i in range(num_patches)]
            
        dict_pat = [(i, j) for i, j in zip(listID, rand_patches)]
        dict_pat_l = list(dict_pat)
        dict_pat_all = dict_pat_all + dict_pat_l
        
    return dict_pat_all       
    


if __name__ in "__main__":

    ###########################################################################################
    # Input section
    ###########################################################################################

    # Run info general selection
    clf_2D    = 'gap'
    models2D  = ['CNNflexi'] 
    

    # Data folders - indicate the relevant folders containing data

    dataFoldersAll = path
    
    # Output folder
    outputtensorboard = path + r'\code\tensorflowCode\outputTF'
    Path(outputtensorboard).mkdir(parents=True, exist_ok=True)
    outputs = [path + r'\code\tensorflowCode\outputTF\anja\output_2D']
    

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', nargs='+', type=int, default=[1])
    parser.add_argument('--patch', nargs='+', type=int, default=[99])
    # Args for hyperparameter screening
    parser.add_argument('--layers', nargs='+', type=int, default=[3])
    parser.add_argument('--kernel_size', nargs='+', type=int, default=[3])
    parser.add_argument('--l1', nargs='+', type=float, default=[1e-05])
    parser.add_argument('--dropout', nargs='+', type=float, default=[0.25])
    parser.add_argument('--dense1', nargs='+', type=int, default=[64])
    parser.add_argument('--dense2', nargs='+', type=int, default=[32])
    parser.add_argument('--n_filters_1', nargs='+', type=int, default=[32])
    parser.add_argument('--n_filters_2', nargs='+', type=int, default=[64])
    parser.add_argument('--l2', nargs='+', type=float, default=[1e-04])
    parser.add_argument('--l1_act', nargs='+', type=float, default=[0])
    parser.add_argument('--hypind', nargs='+', type=int, default=[0]) # Hyperparameter screening index to save accordingly
    # parser.add_argument('--BNloc', nargs = '+', type=int, default=[2]) # BNloc_2D = 2
    parser.add_argument('--BNloc_2D', nargs = '+', type=int, default=[1])
    parser.add_argument('--lr', nargs = '+', type=float, default=[1e-4])
    parser.add_argument('--bs', nargs = '+', type=int, default=[50])
                        
    args      = parser.parse_args()
    thisCV    = args.cv[0] 
    patch_in  = args.patch[0] 
    n_layers_2D = args.layers[0]
    k_size_2D = args.kernel_size[0]
    l1_tail = args.l1[0]
    dropout = args.dropout[0]
    dense1 = args.dense1[0]
    dense2 = args.dense2[0]
    n_filter_1_2D = args.n_filters_1[0]
    n_filter_2_2D = args.n_filters_2[0]
    l2_2D = args.l2[0]
    l1_act2D = args.l1_act[0]
    hypind = args.hypind[0]
    lr = args.lr[0]
    BNloc_2D = args.BNloc_2D[0]
    bs = args.bs[0]
    
    # set number of runs
    runIDsuse = [0,1,2]
    model_ind = 0
    clfind    = 0
    folderID  = 0

    # define Image (patch) dimensions
    dim2D   = [30,30]

    for cv in [thisCV]:

        # Model choices
        clf             = 'gap'
        choice_model2D  = models2D[model_ind]
        modeluse        = choice_model2D + '_' + clf
        print('Model use: ' + modeluse)
        print('CV: ' + str(cv))
        NAME = modeluse+'_'+ '_hypind_' + str(hypind) + '_{}'.format(int(time.time()))
        if model_ind == 1:
            loadParameters = True
            useTensorBoard = False


        output          = os.path.join(outputs[folderID])
        data_folder_2D  = os.path.join(path, dataFoldersAll)

        # Create folders if needed
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_distance')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_eval')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_hdf5')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_losses')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_overviews')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'run_parameters')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output, 'figures')).mkdir(parents=True, exist_ok=True)
        loadfrom = output
        print('Output: ' + output)


        
        dim0p    = dim2D[folderID]
        dim1p    = dim2D[folderID]
        patchdim = (dim0p, dim1p)
        n_channels_2D = 1


        

        ###########################################################################################
        # Run model
        ###########################################################################################

        # Configure GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        #####################################
        #       Get hyperparameters         #
        #####################################

        # set patience as we use early stopping
        pat = 50
        
        # define hyperparameters not specified as args
        n_classes = 2
        n_epochs = 1000
        useDO_2D = False
        l1_den2D = 0.001 
        

        
        input_Combi = {'clf': clf,
                        'n_classes': n_classes,
                        'n_epochs': n_epochs,
                        'bs': bs,
                        'lr': lr,
                        'pat': pat,
                        'l1': l1_tail,
                        'dense1': dense1,
                        'dense2': dense2}


        #####################################
        #              LOAD DATA            #
        #####################################
        # load partitions for 5-fold CV
        partition_file   = path +  str(cv) + '.npy'
        label_file       = path + str(cv) + '.npy'
        partition        = np.load(partition_file, allow_pickle='TRUE').item()  # IDs
        labels           = np.load(label_file, allow_pickle='TRUE').item()  # Labels
        


        # Load 2D image data FROM DATA LOADER
        if choice_model2D != 'None':

            print('Loading 2D MRI data')

            # Data set
            params_dataloader = {'dim': patchdim,
                          'batch_size': input_Combi['bs'],
                          'n_classes': n_classes,
                          'n_channels': 1,
                          'shuffle': True,
                          'datadir': data_folder_2D,
                          }
            params_dataloaderTest = {'dim': patchdim,
                                 'batch_size': input_Combi['bs'],
                                 'n_classes': n_classes,
                                 'n_channels': 1,
                                 'shuffle': False,
                                 'datadir': data_folder_2D,
                                }
            
            
            new_part_trn = load_patches(partition['train'],num_patches, runIDsuse)
            new_part_val = load_patches(partition['validation'], num_patches, runIDsuse)
            new_part_tst = load_patches(partition['test'], num_patches, runIDsuse)
            
            
            
            training_generator = DataGenerator(new_part_trn, labels['train'], **params_dataloader)
            validation_generator = DataGenerator(new_part_val[0:np.floor(len(new_part_val) / params_dataloaderTest['batch_size']).astype(int) * params_dataloaderTest['batch_size']], labels['validation'],**params_dataloaderTest)
            test_generator = DataGenerator(new_part_tst[0:np.floor(len(new_part_tst) / params_dataloaderTest['batch_size']).astype(int) * params_dataloaderTest['batch_size']], labels['test'], **params_dataloaderTest)
                                                             


        y_tst = labels['test']
        y_val = labels['validation']
        y_trn = labels['train']
        
        X_trn = partition['train']
        X_val = partition['validation']
        X_tst= partition['test']
        
        
        
        
       
        # Sample shapes
        sample_shape_2D = (input_Combi['bs'], dim0p, dim1p,  1)

        # hyperparameters
        params_2D = {'n_channels': n_channels_2D,
                     'datadir': data_folder_2D,
                     'patchdim': patchdim,
                     'BNloc': BNloc_2D,
                     'useDO': useDO_2D,
                     'n_filters_1': n_filter_1_2D,
                     'n_filters_2': n_filter_2_2D,
                     'kernel_size': k_size_2D,
                     'l2': l2_2D,
                     'l1_den': l1_den2D,
                     'l1_act': l1_act2D,
                     'sample_shape': sample_shape_2D,
                     'clf': clf_2D,
                     'n_layers': n_layers_2D,
                     'dropout': dropout
                     }



        # Loop over all runs - option to skip
        for counter, runIDuse in enumerate(runIDsuse):

            # Random seeds for reproducible results
            seed(runIDuse)
            tf.random.set_seed(runIDuse)

            # Name
            name = modeluse  + '_run' + str(runIDuse) + 'CV_' + str(cv) \
                   + 'hypind_' + str(hypind)


            #####################################
            #           build the model         #
            #####################################

            # Model input
            input2D    = {'sample_shape': sample_shape_2D,
                       'params': params_2D}
            input      = X_trn
            val_data   = (X_val, y_val)
            val_data_X = X_val
            tst_data   = (X_tst, y_tst)
            tst_data_X = X_tst

            
            # Get the model
            model = CNN_Model(input2D, input_Combi, choice_model2D)

            # Compile the model
            monitor_var     = 'loss'
            monitor_val_var = 'val_loss'
            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=input_Combi['lr']),
                          metrics=None)
            model.summary()
           

            #  Early stopping and check points
            
            filepath      = os.path.join(os.path.join(output, 'run_hdf5'), name+'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_nnet_run_test.hdf5')
            check         = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                               save_best_only=True, mode='auto')
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=input_Combi['pat'],
                                                             mode='auto')
            tensorboard = TensorBoard(log_dir=os.path.join(outputtensorboard, 'logs/{}'.format(NAME)))
            callbacks = [check, earlyStopping, tensorboard]

            # Class weights
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_trn), y_trn)
            class_weight_dict = dict(enumerate(class_weights))

            # Actual training
            history = model.fit(training_generator,
                                validation_data = validation_generator,
                                batch_size=input_Combi['bs'],
                                epochs=input_Combi['n_epochs'],
                                verbose=1,
                                callbacks=callbacks,
                                class_weight=class_weight_dict,
                                shuffle=True)

            # Load the best weights
            model.load_weights(filepath)
            

            # Save model
            filepath_model = os.path.join(os.path.join(output, 'run_hdf5'), 'MODEL_'+ name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_' + str(bs)+ '_nnet_run_test.h5')
            model.save(filepath_model)

            # Get encoding
            encoder = Model(model.input, model.layers[-2].output)
            encoder.summary()

            
            weights = model.layers[-1].get_weights()

            
            filepath_enc_weights = os.path.join(os.path.join(loadfrom, 'run_hdf5'), 'encoding_weights_preClassLayer' + name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs) + '.npy')
            with open(filepath_enc_weights, 'wb') as f:
                np.save(f, weights, allow_pickle=True)

            
            plot_trainingProcess(history, os.path.join(output, 'run_losses'), name,lr,k_size_2D,BNloc_2D,dropout,n_layers_2D,cv)
           
            
            # Evaluate training data, save
            tn_trn, fp_trn, fn_trn, tp_trn, acc_trn, precision_trn, recall_trn, roc_auc_trn, aps_trn, dist_trn, \
            meanDist_trn, stdDist_trn, thresh_opt_trn, y_pred_trn = \
                evalBinaryClassifierCNN(model,training_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation training of ' + modeluse + '_' + clf,
                                        doPlots=True,
                                        batchsize=input_Combi['bs'],
                                        filename='trn_'+name+'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D) + '_bs_' + str(bs))
            dic_dist_trn = {'y_pred': y_pred_trn, 'dist': dist_trn, 'thresh': thresh_opt_trn}
            df_dist_trn = pd.DataFrame(dic_dist_trn)


            # Evaluate validation data, save
            tn, fp, fn, tp, acc, precision, recall, roc_auc, aps, dist, meanDist, stdDist, thresh_opt, y_pred = \
                evalBinaryClassifierCNN(model,validation_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation validation of ' + modeluse + '_' + clf,
                                        doPlots=True,
                                        batchsize=input_Combi['bs'],
                                        filename='val_' + name+'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs),
                                        thresh_opt = thresh_opt_trn)
            dic_dist_val = {'y_pred': y_pred, 'dist': dist, 'thresh': thresh_opt}
            df_dist_val = pd.DataFrame(dic_dist_val)

            # Evaluate test data, save
            tn_tst, fp_tst, fn_tst, tp_tst, acc_tst, precision_tst, recall_tst, roc_auc_tst, aps_tst, dist_tst,\
            meanDist_tst, stdDist_tst, thresh_opt_tst, y_pred_tst = \
                evalBinaryClassifierCNN(model,test_generator, os.path.join(output, 'run_eval'),
                                        labels=['Negatives', 'Positives'],
                                        plot_title='Evaluation Test of ' + modeluse + '_' + clf,
                                        doPlots=True,
                                        batchsize=input_Combi['bs'],
                                        filename='test_' + name+'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D) + '_bs_' + str(bs),
                                        thresh_opt=thresh_opt_trn)
            dic_dist_tst = {'y_pred': y_pred_tst, 'dist': dist_tst, 'thresh': thresh_opt_tst}
            df_dist_tst = pd.DataFrame(dic_dist_tst)


            # Save results to df
            data_df = {'runID': str(runIDuse),
                       'data_folder': data_folder_2D,
                       'model': modeluse,
                       'clf': clf,
                       'n_epochs': n_epochs,
                       'tn': tn,
                       'fp': fp,
                       'fn': fn,
                       'tp': tp,
                       'acc': acc,
                       'precision': precision,
                       'recall': recall,
                       'auc': roc_auc,
                       'aps': aps,
                       'meanDist': meanDist,
                       'stdDist': stdDist
                       }
            df = pd.DataFrame(data=data_df, index=[runIDuse])
           
            df.to_csv(os.path.join(os.path.join(output, 'run_overviews'), name + '_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_cv'+str(cv)+'_results_overview.csv'))
           

            data_df_tst = {'runID': str(runIDuse),
                       'data_folder': data_folder_2D,
                       'model': modeluse,
                       'clf': clf,
                       'n_epochs': n_epochs,
                       'tn': tn_tst,
                       'fp': fp_tst,
                       'fn': fn_tst,
                       'tp': tp_tst,
                       'acc': acc_tst,
                       'precision': precision_tst,
                       'recall': recall_tst,
                       'auc': roc_auc_tst,
                       'aps': aps_tst,
                       'meanDist': meanDist_tst,
                       'stdDist': stdDist_tst
                       }
            df_tst = pd.DataFrame(data=data_df_tst, index=[runIDuse])
            df_tst.to_csv(os.path.join(os.path.join(output, 'run_overviews'), name + '_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_c'+str(cv)+'_results_overview_tst.csv'))
          
            # Save run distance to decision boundary
            df_dist_trn.to_csv(os.path.join(os.path.join(output, 'run_distance'), name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs)+ '_results_dist_trn.csv'))
            df_dist_val.to_csv(os.path.join(os.path.join(output, 'run_distance'), name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs)+'_results_dist.csv'))
            df_dist_tst.to_csv(os.path.join(os.path.join(output, 'run_distance'), name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs)+'_results_dist_tst.csv'))

            # Save parameters
            file = os.path.join(os.path.join(output, 'run_parameters'), name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs)+'_param3D.pkl')
           
            with open(file, 'wb') as f:
                pickle.dump(params_2D, f)

          
            file = os.path.join(os.path.join(output, 'run_parameters'), name +'_lr_' + str(lr) + '_KS_' + str(k_size_2D) + '_BN_' + str(BNloc_2D) + '_dp_' + str(dropout) + '_lay_' + str(n_layers_2D)+'_bs_'+str(bs)+ '_paramCombi.pkl')
                             
            with open(file, 'wb') as f:
                pickle.dump(input_Combi, f)


            # Clear keras
            tf.keras.backend.clear_session()


