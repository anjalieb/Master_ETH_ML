###########################################################################################
# Imports
###########################################################################################

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, average_precision_score



###########################################################################################
# Functions
###########################################################################################

def evalBinaryClassifierCNN(model,training_generator,mydir, labels=['Positives', 'Negatives'],
                         plot_title='Evaluation of model', doPlots=True, batchsize = 50,
                        filename = '', thresh_opt=None):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.

    Displays a labelled Confusion Matrix, distributions of the predicted
    probabilities for both classes, the Recall-Precision curve, the ROC curve, and F1 score of a fitted
    Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

    Parameters
    ----------
    model : fitted scikit-learn model with predict_proba & predict methods
        and classes_ attribute. Typically LogisticRegression or
        LogisticRegressionCV

    x : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples
        in the data to be tested, and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target vector relative to x.

    labels: list, optional
        list of text labels for the two classes, with the positive label first

    Displays
    ----------
    4 Subplots

    Returns
    ----------
    F1: float
    '''

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    x,y = training_generator.__getall__()
    results = model.evaluate(x, batch_size=batchsize)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`

    y_pred = model.predict(x)[:,1]


    # Optimize the threshold:
    # if thresh_opt is None:
    #     F1_opt = 0
    #     thresh_opt = 0
    #     for t in np.arange(0,1,0.01):
    #         y_pred_class  = np.where(y_pred > t, 1,0)
    #         cm = confusion_matrix(y[:, 1], y_pred_class)
    #         tn, fp, fn, tp = [i for i in cm.ravel()]
    #         precision = tp / (tp + fp)
    #         recall = tp / (tp + fn)
    #         F1 = 2 * (precision * recall) / (precision + recall)
    #         if F1>F1_opt:
    #             thresh_opt = t
    #             F1_opt = F1
    
    # Option: set thresh_opt to 0.5 without optimizing it
    thresh_opt = 0.5

    y_pred_class = np.where(y_pred > thresh_opt, 1, 0)

    # get mean realtive distance to descision boundary
    dist = []
    for this_y in list(y_pred):
        if this_y-thresh_opt>0:
            distance = (this_y-thresh_opt)/(1-thresh_opt)
        else:
            distance = -(this_y - thresh_opt) / (thresh_opt)
        dist.append(distance)
    meanDist = np.mean(dist)
    stdDist = np.std(dist)



    # model predicts probabilities of positive class
    # p = model.predict_proba(x)
    # if len(model.classes_) != 2:
    #     raise ValueError('A binary class problem is required')
    # if model.classes_[1] == 1:
    #     pos_p = p[:, 1]
    #     pos_ind = 0
    # elif model.classes_[0] == 1:
    #     pos_p = p[:, 0]
    #     pos_ind = 0

    # 1 -- Confusion matrix
    cm = confusion_matrix(y[:,1], y_pred_class)

    # 2 -- Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probPos': np.squeeze(y_pred), 'target': y[:,1]})

    # 3 -- PRC:
    #y_score = model.predict_proba(x)[:, pos_ind]
    precision, recall, _ = precision_recall_curve(y[:,1], y_pred, pos_label=1)
    aps = average_precision_score(y[:,1], y_pred)

    # 4 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y[:,1], y_pred, pos_label=1)
    roc_auc = auc(fp_rates, tp_rates)
    tn, fp, fn, tp = [i for i in cm.ravel()]

    # FIGURE
    if doPlots:
        fig = plt.figure(figsize=[20, 5])
        fig.suptitle(plot_title, fontsize=16)

        # 1 -- Confusion matrix
        plt.subplot(141)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                         annot_kws={"size": 14}, fmt='g')
        cmlabels = ['True Negatives', 'False Positives',
                    'False Negatives', 'True Positives']
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title('Confusion Matrix', size=15)
        plt.xlabel('Predicted Values', size=13)
        plt.ylabel('True Values', size=13)

        # 2 -- Distributions of Predicted Probabilities of both classes
        plt.subplot(142)
        plt.hist(df[df.target == 1].probPos, density=True, bins=25,
                 alpha=.5, color='green', label=labels[0])
        plt.hist(df[df.target == 0].probPos, density=True, bins=25,
                 alpha=.5, color='red', label=labels[1])
        plt.axvline(thresh_opt, color='blue', linestyle='--', label='Boundary')
        plt.xlim([0, 1])
        plt.title('Distributions of Predictions', size=15)
        plt.xlabel('Positive Probability (predicted)', size=13)
        plt.ylabel('Samples (normalized scale)', size=13)
        plt.legend(loc="upper right")

        # 3 -- PRC:
        plt.subplot(143)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.title('Recall-Precision Curve', size=15)
        plt.text(0.1, 0.3, f'AURPC = {round(aps, 2)}')
        plt.xlabel('Recall', size=13)
        plt.ylabel('Precision', size=13)
        plt.ylim([0.2, 1.05])
        plt.xlim([0.0, 1.0])
        # #plt.show(block=False)

        # 4 -- ROC curve with annotated decision point
        plt.subplot(144)
        plt.plot(fp_rates, tp_rates, color='green',
                 lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

        # plot current decision point:
        plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', size=13)
        plt.ylabel('True Positive Rate', size=13)
        plt.title('ROC Curve', size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=.3)
        #plt.show(block=False)

        filenameuse = os.path.join(mydir, plot_title+filename + '.png')
        plt.savefig(filenameuse)
        plt.close('all')

    # 5 -- F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    # 6 -- Accuracy:
    acc = (tp + tn) / (tp + tn + fp + fn)



    # Print
    printout = (
        f'Precision: {round(precision, 2)} | '
        f'Recall: {round(recall, 2)} | '
        f'F1 Score: {round(F1, 2)} | '
        f'Accuracy: {round(acc, 2)} | '
    )
    print(printout)

    return tn, fp, fn, tp,acc, precision,recall,roc_auc,aps, dist,meanDist,stdDist, thresh_opt, y_pred



def plot_trainingProcess(history,output,name, lr ,kernel_size, BNloc_2D, dropout, n_layers_2D, cv):


    nameUse = 'loss_epoch_'+ '_LR_' + str(lr) + '_KS_' + str(kernel_size)+ '_BNloc_' + str(BNloc_2D) + '_dp_' + str(dropout)+ '_lay_' + str(n_layers_2D) + '_cv_' + str(cv)+ name + '.png'
    filename = os.path.join(output, nameUse)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.savefig(filename)
    #plt.show(block=False)
    plt.close('all')

   

    try:
        nameUse = 'acc_epoch_' + name + '.png'
        filename = os.path.join(output, nameUse)
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(filename)
        #plt.show(block=False)
        plt.close('all')
    except:
        print('No accuracy')

    try:
        print(history.history.keys())
        nameUse = 'auc_epoch_' + name + '.png'
        filename = os.path.join(output, nameUse)
        plt.figure()
        try:
            plt.plot(history.history['auc'])
            plt.plot(history.history['val_auc'])
        except:
            plt.plot(history.history['auc_'+str(counter)])
            plt.plot(history.history['val_auc_'+str(counter)])
        plt.title('Model auc')
        plt.ylabel('AUC')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(filename)
        #plt.show(block=False)
        plt.close('all')
    except:
        print('No auc')

    return 0


