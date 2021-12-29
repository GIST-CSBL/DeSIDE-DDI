import pandas as pd
import numpy as np

import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import math

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats

# find feature in the generator
def find_exp(drug_df, ts_exp, column_name):
    return pd.merge(drug_df, ts_exp, left_on=column_name, right_on='pubchem', how='left').iloc[:,2:]

# Generator
class custom_dataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_label, batch_size, exp_df, shuffle=True):
        self.x = x_set
        self.y = y_label
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))
        self.shuffle = shuffle
        self.exp_df = exp_df
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return math.ceil(len(self.x)/self.batch_size)
        
    def __data_generation__(self, x_list):
        x1 = find_exp(x_list[['drug1']], self.exp_df, 'drug1')
        x2 = find_exp(x_list[['drug2']], self.exp_df, 'drug2')
        x_se = x_list['SE']
        
        x_se_one_hot = to_categorical(x_list['SE'], num_classes=963)

        x1 = np.array(x1).astype(float)
        x2 = np.array(x2).astype(float)
        
        return x1, x2, x_se, x_se_one_hot
        
    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x.iloc[indexes]
        batch_y = self.y[indexes]        
        
        x1, x2, x_se, x_se_one_hot = self.__data_generation__(batch_x)
                
        return [x1, x2, x_se, x_se_one_hot], batch_y
    
# =============================================================================================
# Model settings
# =============================================================================================

# Checkpoint
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self, save_path, model_name, init_learining_rate, decay_rate, decay_steps, \
                 save_best_metric='val_loss',this_max=False, **kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.epoch_loss = {}
        self.epoch_val_loss = {}
        self.save_path = save_path
        self.model_name = model_name
        
        self.init_learining_rate = init_learining_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = 0
        
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')
        
    def on_epoch_end(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
#         print('learning rate: %.5f'%lr)
        
        metric_value = logs.get(self.save_best_metric)
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_model = self.model
        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_model = self.model
                
        self.epoch_loss[epoch] = logs.get('loss')
        self.epoch_val_loss[epoch] = logs.get('val_loss')
        self.best_model.save_weights(self.save_path + self.model_name + '.h5')
        
    def on_epoch_begin(self, epoch, logs={}):
        actual_lr = float(K.get_value(self.model.optimizer.lr))
        decayed_learning_rate = actual_lr * self.decay_rate ** (epoch / self.decay_steps)
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        if epoch % 10 == 0:
            K.set_value(self.model.optimizer.lr, self.init_learining_rate)


# =============================================================================================
# Model Evaluation
# =============================================================================================

def mean_predicted_score(true_df, predicted_y, with_plot=True):
    test_pred_result = pd.concat([true_df.reset_index(drop=True), pd.DataFrame(predicted_y, columns=['predicted_score'])], axis=1)
    
    if (with_plot):
        fig, ax = plt.subplots(figsize=(6,6))
        temp = test_pred_result.groupby('label')['predicted_score'].apply(list)
        sns.boxplot(x='label', y='predicted_score', data=test_pred_result[['label','predicted_score']])
        plt.show()
    
    return test_pred_result

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def cal_performance(predicted_scores_df):
    uniqueSE = predicted_scores_df.SE.unique()

    dfDict = {elem : pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = predicted_scores_df[:][predicted_scores_df.SE == key]
        
    se_performance = pd.DataFrame(columns=['Side effect no.','median_pos', 'median_neg', 'optimal_thr','SN','SP','PR','AUC','AUPR'])
    for se in uniqueSE:
        df = dfDict[se]
        
        med_1 = np.median(df[df.label == 1.0].predicted_score)
        med_0 = np.median(df[df.label == 0.0].predicted_score)

        temp_thr = (med_1 + med_0)/2
        temp_y = df.predicted_score.apply(lambda x: 0 if x > temp_thr else 1)
        tn, fp, fn, tp = confusion_matrix(df.label, temp_y).ravel()

        optimal_thr = Find_Optimal_Cutoff(1-df.label, df.predicted_score)[0]
        temp_y_opt = df.predicted_score.apply(lambda x: 0 if x > optimal_thr else 1)
        tn, fp, fn, tp = confusion_matrix(df.label, temp_y_opt).ravel()

        auc = roc_auc_score(1-df.label, df.predicted_score)
        aupr = average_precision_score(1-df.label, df.predicted_score)

        temp_df = pd.DataFrame({'Side effect no.':se, 'median_pos':med_1, 'median_neg':med_0, 'optimal_thr':optimal_thr, \
                                'SN':tp/(tp+fn), 'SP':tn/(tn+fp), 'PR':tp/(tp+fp), 'AUC':auc, 'AUPR':aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)
        
    return se_performance

def calculate_test_performance(predicted_scores_df):
    uniqueSE = predicted_scores_df.SE.unique()

    dfDict = {elem : pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = predicted_scores_df[:][predicted_scores_df.SE == key]
        
    se_performance = pd.DataFrame(columns=['Side effect no.','SN','SP','PR','AUC','AUPR'])
    for se in uniqueSE:
        df = dfDict[se]

        tn, fp, fn, tp = confusion_matrix(df.label, df.predicted_label).ravel()

        auc = roc_auc_score(1-df.label, df.predicted_score)
        aupr = average_precision_score(1-df.label, df.predicted_score)

        temp_df = pd.DataFrame({'Side effect no.':se, \
                                'SN':tp/(tp+fn), 'SP':tn/(tn+fp), 'PR':tp/(tp+fp), 'AUC':auc, 'AUPR':aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)
        
    return se_performance

def calculate_predicted_label_ver3(predicted_score_df, optimal_thr, se_col_name='SE', threshold_col_name='optimal_thr'):
    # 1) 마지막 5개 값 평균
    temp_thr = pd.DataFrame(optimal_thr.iloc[:, -7:-2].mean(axis=1), columns=[threshold_col_name])
    
    thr = pd.concat([optimal_thr['SE'], temp_thr], axis=1)
    
    merged = pd.merge(predicted_score_df, thr, left_on='SE', right_on=se_col_name, how='left')
    merged['predicted_label'] = merged['predicted_score'] < merged[threshold_col_name]
    merged.predicted_label = merged.predicted_label.map(int)
    merged['gap'] = merged['predicted_score'] - merged[threshold_col_name]
    merged.gap = merged.gap.map(abs)
    test_perf = merged[['drug1','drug2','SE','label','predicted_label','predicted_score','gap']]
    return test_perf, thr

def external_validation_v2(model, test_x, test_y, exp_df, optimal_threshold, batch_size):
    test_gen = custom_dataGenerator(test_x, test_y.values, batch_size=batch_size, exp_df=exp_df, shuffle=False)
    pred_test = model.predict_generator(generator=test_gen)
    
    test_prediction_scores = mean_predicted_score(pd.concat([test_x, test_y], axis=1), pred_test)
    test_prediction_predicted_label_df, thr = calculate_predicted_label_ver3(test_prediction_scores, optimal_threshold)
    test_prediction_perf_df = calculate_test_performance(test_prediction_predicted_label_df)
    
    return test_prediction_predicted_label_df, test_prediction_perf_df, thr

# Calculate average predicted scores & relabel
def merge_both_pairs(ori_predicted_label_df, swi_predicted_label_df, optimal_threshold, thr_col_name):
    merge_label = pd.merge(ori_predicted_label_df, swi_predicted_label_df, left_on=['drug1','drug2','SE'], right_on=['drug2','drug1','SE'])[['drug1_x','drug2_x','SE','label_x','predicted_label_x','predicted_label_y', 'predicted_score_x', 'predicted_score_y']]
    merge_label['mean_predicted_score'] = (merge_label.predicted_score_x + merge_label.predicted_score_y)/2
    merge_label.rename(columns={'drug1_x':'drug1','drug2_x':'drug2', 'SE_x':'SE','label_x':'label'}, inplace=True)
    
    merged = pd.merge(merge_label, optimal_threshold, left_on='SE', right_on='SE', how='left')
    merged['final_predicted_label'] = merged['mean_predicted_score'] < merged[thr_col_name]
    merged.final_predicted_label = merged.final_predicted_label.map(int)
    merged['gap'] = merged['mean_predicted_score'] - merged[thr_col_name]
    merged.gap = merged.gap.map(abs)
    
    merged = merged[['drug1','drug2','SE','label','predicted_label_x','predicted_label_y', 'predicted_score_x', 'predicted_score_y', \
            'mean_predicted_score','final_predicted_label','gap']]
    
    #======================================================================================
    uniqueSE = merged.SE.unique()

    dfDict = {elem : pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = merged[:][merged.SE == key]
        
    se_performance = pd.DataFrame(columns=['Side effect no.','SN','SP','PR','AUC','AUPR'])
    for se in uniqueSE:
        df = dfDict[se]

        tn, fp, fn, tp = confusion_matrix(df.label, df.final_predicted_label).ravel()

        auc = roc_auc_score(1-df.label, df.mean_predicted_score)
        aupr = average_precision_score(1-df.label, df.mean_predicted_score)

        temp_df = pd.DataFrame({'Side effect no.':se, \
                                'SN':tp/(tp+fn), 'SP':tn/(tn+fp), 'PR':tp/(tp+fp), 'AUC':auc, 'AUPR':aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)
        
    return merged, se_performance
