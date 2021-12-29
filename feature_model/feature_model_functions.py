import pandas as pd
import numpy as np
import math

def split_dataset(df):
    x_fp = df.filter(regex='fp')
    y_label = df.drop(x_fp, axis=1)
    
    return x_fp, y_label

def split_train_test(x):
    x = x.sample(frac=1)
    
    train = round(x.shape[0]*0.8)
    print(train)
    
    return x.iloc[0:train,:], x.iloc[train:,:]

def split_dataset_descriptor(df):
    x_fp = df.filter(regex='fp')
    temp = df.drop(x_fp, axis=1)
    x_desc = temp.iloc[:,0:100]
    y_label = temp.iloc[:,100:]

    return x_desc, y_label

def split_dataset_descriptor_both(df):
    x_fp = df.filter(regex='fp')
    temp = df.drop(x_fp, axis=1)
    x_desc = temp.iloc[:,0:100]
    y_label = temp.iloc[:,100:]

    return x_fp, x_desc, y_label

def split_features(df):
    x_fp = df.filter(regex='fp')
    temp = df.drop(x_fp, axis=1)
    x_desc = temp
    
    return x_fp, x_desc