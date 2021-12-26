import pandas as pd
import numpy as np

def load_data(file_path = '/DAS_Storage3/eykim/DDI/202011/ddi_se_dataset_all/'):

    train_x = pd.read_csv(file_path+'train_x.csv', index_col=0)
    train_y = pd.read_csv(file_path+'train_y.csv', index_col=0)
    
    train_data = pd.concat([train_x, pd.DataFrame(train_y, columns=['label'])], axis=1)

    switch_pair = train_x[['drug2','drug1','SE']]
    switch_pair.columns = ['drug1','drug2','SE']
    switch_df = pd.concat([switch_pair, pd.DataFrame(train_y, columns=['label'])], axis=1)

    double_train_data = pd.concat([train_data, switch_df], axis=0)
    print('Including reverse pairs: ', double_train_data.shape)
    
    test_x = pd.read_csv(file_path+'test_x.csv', index_col=0)
    test_y = pd.read_csv(file_path+'test_y.csv', index_col=0)
    
    return double_train_data, test_x, test_y

def load_exp(file_path='/DAS_Storage1/NAS_31/users/eunyoung/Compound_Combination/202008/'):
    return pd.read_csv(file_path + 'ts_predicted_exp_Scaled.csv', index_col=0)
