#!/usr/bin/env python
# coding: utf-8

# In[118]:

    

#Get all functions from code_skeleton
from code_skeleton import *

# Since the processing of the dataframe is vering time intensive, we'll save the final dataframes into a compressed csv format

#df_training_final=pickle.load(open('../Code/saved_variables/df_training_final', 'rb'))
#df_training_final=pickle.load(open('../Code/saved_variables/df_training_final', 'rb'))
#df_testing_final=pickle.load(open('../Code/saved_variables/df_testing_final', 'rb'))
# # Part 1: Run the data processing pipeline on the training dataset and on the test dataset

# In[26]:


# data processing pipeline on the training dataset

dic_organised=organise_data(data_train)
df_engineered=extract_features(dic_organised)
features_to_keep=dropping_features(df_engineered, 0.3)
df_training_final=prepare_features(features_to_keep, df_engineered)


# In[27]:


# data processing pipeline on the testing dataset
dic_organised_test=organise_data(data_test)
df_engineered_test=extract_features(dic_organised_test)
df_testing_final=prepare_features(features_to_keep, df_engineered_test)


# # Part 2: Training on the classification subtask
# 
# ## 2.2 Sepsis prediction
# 

# In[16]:


df_train, df_test, df_labels, labels_test=subset_splitting(df_training_final, labels_train)


# We predicted subtasks 1 and 2 through the same framework, by training a LogisticRegression and a Linear SVM with different parameters on our features and exploring the regularisation parameter space using a GridSearchCV object.
# We saved the GridSearchCV objects for reference in a folder ./GridSearchResults using the pickle module under GridSearchResults.
# 
# Looking at those results, we have checked that the 'best estimator' as defined by GridSearchCV gave acceptable results both on average on Cross Validation and on the validation set (df_test). We then selected for each label the estimator which performed best on the validation set, and used it for our final predictions.

# In[ ]:


# Subtasks 1 and 2:

labels_to_predict=['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                   'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                   'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

def get_best_models(labels_list):
    for label in labels_list:
        best_models={}
        print(label)
        best_models[label]=find_best_model(label, df_train, df_labels)
    return(best_models)

    


# In[26]:


best_models=get_best_models(labels_to_predict)


# **Reload all the GridSearchCV objects to study them**

# In[ ]:

    # This is assuming that get_best_models have been run and has saved the models using pickle

# Logistic Regression Model
clf_log_Alkalinephos= pickle.load(open('GridSearchResults/clf_log_LABEL_Alkalinephos', 'rb'))
clf_log_AST= pickle.load(open('GridSearchResults/clf_log_LABEL_AST', 'rb'))
clf_log_BaseExcess= pickle.load(open('GridSearchResults/clf_log_LABEL_BaseExcess', 'rb'))
clf_log_Bilirubin_direct= pickle.load(open('GridSearchResults/clf_log_LABEL_Bilirubin_direct', 'rb'))
clf_log_Bilirubin_total= pickle.load(open('GridSearchResults/clf_log_LABEL_Bilirubin_total', 'rb'))
clf_log_EtCO2= pickle.load(open('GridSearchResults/clf_log_LABEL_EtCO2', 'rb'))
clf_log_Fibrinogen= pickle.load(open('GridSearchResults/clf_log_LABEL_Fibrinogen', 'rb'))
clf_log_Lactate= pickle.load(open('GridSearchResults/clf_log_LABEL_Lactate', 'rb'))
clf_log_SaO2= pickle.load(open('GridSearchResults/clf_log_LABEL_SaO2', 'rb'))
clf_log_Sepsis= pickle.load(open('GridSearchResults/clf_log_LABEL_Sepsis', 'rb'))
clf_log_TroponinI= pickle.load(open('GridSearchResults/clf_log_LABEL_TroponinI', 'rb'))

# Linear SVM Model
clf_lin_Alkalinephos= pickle.load(open('GridSearchResults/clf_lin_LABEL_Alkalinephos', 'rb'))
clf_lin_AST= pickle.load(open('GridSearchResults/clf_lin_LABEL_AST', 'rb'))
clf_lin_BaseExcess= pickle.load(open('GridSearchResults/clf_lin_LABEL_BaseExcess', 'rb'))
clf_lin_Bilirubin_direct= pickle.load(open('GridSearchResults/clf_lin_LABEL_Bilirubin_direct', 'rb'))
clf_lin_Bilirubin_total= pickle.load(open('GridSearchResults/clf_lin_LABEL_Bilirubin_total', 'rb'))
clf_lin_EtCO2= pickle.load(open('GridSearchResults/clf_lin_LABEL_EtCO2', 'rb'))
clf_lin_Fibrinogen= pickle.load(open('GridSearchResults/clf_lin_LABEL_Fibrinogen', 'rb'))
clf_lin_Lactate= pickle.load(open('GridSearchResults/clf_lin_LABEL_Lactate', 'rb'))
clf_lin_SaO2= pickle.load(open('GridSearchResults/clf_lin_LABEL_SaO2', 'rb'))
clf_lin_Sepsis= pickle.load(open('GridSearchResults/clf_lin_LABEL_Sepsis', 'rb'))
clf_lin_TroponinI= pickle.load(open('GridSearchResults/clf_lin_LABEL_TroponinI', 'rb'))


# In[7]:


#create a list of all loaded variables from above of the Log Regression model
all_LR = [clf_log_Alkalinephos,clf_log_AST,clf_log_BaseExcess,clf_log_Bilirubin_direct,clf_log_Bilirubin_total,clf_log_EtCO2,clf_log_Fibrinogen,clf_log_Lactate,clf_log_SaO2,clf_log_Sepsis,clf_log_TroponinI]
#create a list with all 'LABEL_Alkalinephos' names in the same order as the previous list all_LR
all_LR_names = ['LABEL_Alkalinephos','LABEL_AST', 'LABEL_BaseExcess', 'LABEL_Bilirubin_direct', 'LABEL_Bilirubin_total','LABEL_EtCO2','LABEL_Fibrinogen','LABEL_Lactate','LABEL_SaO2','LABEL_Sepsis','LABEL_TroponinI']
print(all_LR_names)
#create a list of all loaded variables from above of the lin SVC model
all_linSVC = [clf_lin_Alkalinephos,clf_lin_AST,clf_lin_BaseExcess,clf_lin_Bilirubin_direct,clf_lin_Bilirubin_total,clf_lin_EtCO2,clf_lin_Fibrinogen,clf_lin_Lactate,clf_lin_SaO2,clf_lin_Sepsis,clf_lin_TroponinI]
#create a list with all 'LABEL_Alkalinephos' names in the same order as the previous list all_linSVC
all_linSVC_names = ['LABEL_Alkalinephos','LABEL_AST', 'LABEL_BaseExcess', 'LABEL_Bilirubin_direct', 'LABEL_Bilirubin_total','LABEL_EtCO2','LABEL_Fibrinogen','LABEL_Lactate','LABEL_SaO2','LABEL_Sepsis','LABEL_TroponinI']
print(all_linSVC_names)


# In[8]:


# get the best estimator incl C parameter and the best score of ever variable in all_LR
for label,name in zip(all_LR,all_LR_names):
    print(name)
    lr_model = label.best_estimator_
    print("this is the best lr model ")
    print(lr_model)
    best_score_training = label.best_score_
    print("this is the best score ")
    print(best_score_training)
    lr_model.fit(df_train,labels_train[name])
    # We predict the proba of the positive outcome
    prediction=lr_model.predict_proba(df_test)[:,1]
    test_score=roc_auc_score(y_true=labels_test[name],y_score=prediction, average='weighted')
    print("this is the test score ")
    print(test_score)
    # compare the test_score to the best_score (score of how well model performed on training data)
    
for label, name_svc in zip(all_linSVC,all_linSVC_names):
    print(name_svc)
    svc_model = label.best_estimator_
    print("this is the best svc model ")
    print(svc_model)
    best_score_training = label.best_score_
    print("this is the best score ")
    print(best_score_training)
    svc_model.fit(df_train,labels_train[name_svc])
    # We predict the proba of the positive outcome
    prediction=svc_model.predict_proba(df_test)[:,1]
    test_score=roc_auc_score(y_true=labels_test[name_svc],y_score=prediction, average='weighted')
    print("this is the test score ")
    print(test_score)
    


# **We want to get a prediction for every label, apply the model of choice deduced above**
# 
# As the models have selected manually on the basis of what has been printed in the cell above, each model has also been trained individually

# In[9]:


model_dic={}
prediction_dic={}

model_Alkalinephos=LogisticRegression(C=0.001,class_weight='balanced',random_state=5, verbose=True)
model_BaseExcess=LogisticRegression(C=0.1,max_iter=1000,class_weight='balanced',random_state=5, verbose=True )
model_BilirubinDirect=LogisticRegression(C=0.001,class_weight='balanced',random_state=5, verbose=True )
model_BilirubinTotal=LogisticRegression(C=0.1,max_iter=1000,class_weight='balanced',random_state=5, verbose=True )
model_AST=LogisticRegression(C=0.1,max_iter = 1000,class_weight='balanced',random_state=5, verbose=True )
model_EtCO2=LogisticRegression(C=0.1,max_iter = 1000,class_weight='balanced',random_state=5, verbose=True )
model_Fibrinogen = svm.SVC(C=0.01,probability=True, kernel='linear',class_weight='balanced', random_state=5)
model_Lactate = svm.SVC(C=0.1,probability=True, kernel='linear',class_weight='balanced', random_state=5)
model_SaO2 = svm.SVC(C=0.1,probability=True, kernel='linear',class_weight='balanced', random_state=5)
model_Sepsis=LogisticRegression(C=0.001,class_weight='balanced',random_state=5, verbose=True )
model_TroponinI = svm.SVC(C=0.001,probability=True, kernel='linear',class_weight='balanced', random_state=5)


# In[19]:


model_dic['LABEL_Alkalinephos'], prediction_dic['LABEL_Alkalinephos']=train_best_model_and_predict('LABEL_Alkalinephos', model_Alkalinephos, df_train,df_labels, df_testing_final)


# In[20]:


model_dic['LABEL_BaseExcess'], prediction_dic['LABEL_BaseExcess']=train_best_model_and_predict('LABEL_BaseExcess', model_BaseExcess, df_train,df_labels, df_testing_final)


# In[23]:


model_dic['LABEL_Bilirubin_direct'], prediction_dic['LABEL_Bilirubin_direct']=train_best_model_and_predict('LABEL_Bilirubin_direct', model_BilirubinDirect, df_train,df_labels, df_testing_final)


# In[24]:


model_dic['LABEL_AST'], prediction_dic['LABEL_AST']=train_best_model_and_predict('LABEL_AST', model_AST, df_train,df_labels, df_testing_final)


# In[25]:


model_dic['LABEL_Bilirubin_total'], prediction_dic['LABEL_Bilirubin_total']=train_best_model_and_predict('LABEL_Bilirubin_total', model_BilirubinTotal, df_train,df_labels, df_testing_final)


# In[26]:


model_dic[ 'LABEL_EtCO2'], prediction_dic[ 'LABEL_EtCO2']=train_best_model_and_predict( 'LABEL_EtCO2', model_EtCO2, df_train,df_labels, df_testing_final)


# In[27]:


model_dic['LABEL_Fibrinogen'], prediction_dic['LABEL_Fibrinogen']=train_best_model_and_predict( 'LABEL_Fibrinogen', model_Fibrinogen, df_train,df_labels, df_testing_final)


# In[28]:


model_dic['LABEL_Lactate'], prediction_dic['LABEL_Lactate']=train_best_model_and_predict('LABEL_Lactate', model_Lactatectate, df_train,df_labels, df_testing_final)


# In[29]:


model_dic['LABEL_SaO2'], prediction_dic['LABEL_SaO2']=train_best_model_and_predict('LABEL_SaO2', model_SaO2, df_train,df_labels, df_testing_final)


# In[30]:


model_dic['LABEL_Sepsis'], prediction_dic['LABEL_Sepsis']=train_best_model_and_predict('LABEL_Sepsis', model_Sepsis, df_train,df_labels, df_testing_final)


# In[31]:


model_dic['LABEL_TroponinI'], prediction_dic['LABEL_TroponinI']=train_best_model_and_predict('LABEL_TroponinI', model_TroponinI, df_train,df_labels, df_testing_final)


# In[33]:


# Save for reference
pickle.dump(model_dic, open('saved_variables/model_dic', 'wb'))
pickle.dump(prediction_dic, open('saved_variables/prediction_dic', 'wb'))


# ## 2.3: Regression Subtask

# In[34]:


#Pick only the correspondent columns for subtask 3
Train_labels_df=df_labels[['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']].to_numpy()
print(Train_labels_df.shape)


# In[41]:


clf = create_and_train_model(df_train, Train_labels_df)


# **Make predictions**

# In[44]:


pred = clf.predict(df_testing_final)


# In[50]:


pred_task3_df=pd.DataFrame(pred, columns=['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'], index=df_testing_final.index)


# In[63]:


pred_task3_df


# **Save model and predictions**

# In[52]:


pickle.dump(clf, open('saved_variables/subtask3_model', 'wb'))
pickle.dump(pred_task3_df, open('saved_variables/subtask_3_prediction', 'wb'))


# # Part 3: Generate the output file

# In[112]:


# set a column as pid
pid_column=prediction_dic['LABEL_AST'].copy(deep=True)
pid_column['pid']=pid_column.index


# In[113]:


final_output_df=pd.concat([pid_column['pid'],prediction_dic['LABEL_BaseExcess'], prediction_dic['LABEL_Fibrinogen'],
                           prediction_dic['LABEL_AST'], prediction_dic['LABEL_Alkalinephos'],
                           prediction_dic['LABEL_Bilirubin_total'], prediction_dic['LABEL_Lactate'],
                           prediction_dic['LABEL_TroponinI'], prediction_dic['LABEL_SaO2'],
                           prediction_dic['LABEL_Bilirubin_direct'], prediction_dic['LABEL_EtCO2'],
                           prediction_dic['LABEL_Sepsis'], pred_task3_df['LABEL_RRate'],
                           pred_task3_df['LABEL_ABPm'], pred_task3_df['LABEL_SpO2'],
                           pred_task3_df['LABEL_Heartrate']], axis=1)


# In[114]:


final_output_df


# In[115]:


final_output_df.to_csv('prediction_output.zip', index=False, float_format='%.3f', compression='zip')

