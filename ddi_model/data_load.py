import pandas as pd
import numpy as np

# input file names (CSV formatted)
def load_data(file_path = '../data/', train_x, train_y, test_x, test_y):
    
    train_x = pd.read_csv(file_path+train_x, index_col=0)
    train_y = pd.read_csv(file_path+train_y, index_col=0)
    
    train_data = pd.concat([train_x, pd.DataFrame(train_y, columns=['label'])], axis=1)

    switch_pair = train_x[['drug2','drug1','SE']]
    switch_pair.columns = ['drug1','drug2','SE']
    switch_df = pd.concat([switch_pair, pd.DataFrame(train_y, columns=['label'])], axis=1)

    double_train_data = pd.concat([train_data, switch_df], axis=0)
    print('Including reverse pairs: ', double_train_data.shape)
    
    test_x = pd.read_csv(file_path+test_x, index_col=0)
    test_y = pd.read_csv(file_path+test_y, index_col=0)
    
    return double_train_data, test_x, test_y

def load_exp(file_path='../data/'):
    return pd.read_csv(file_path + 'twosides_predicted_expression_scaled.csv', index_col=0)
