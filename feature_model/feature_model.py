from feature_model_functions import *

import tensorflow as tf
import keras
from keras.layers import *
from keras.regularizers import *
import keras.backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import metrics


class Feature_model(object):
    def __init__(self,struct_only=False, property_only=False):

        self.struct_only = struct_only
        self.property_only = property_only
        
        self.structure_dim = 1024
        self.property_dim = 100
        self.callbacks = []
        self.model = self.build()

    def build(self):
        
        if (self.struct_only):
            input_drug = Input(shape=(self.structure_dim,))

            hidden = Dense(512, input_dim=self.structure_dim, activation='relu', kernel_regularizer=l2(0.01))(input_drug)
            hidden = BatchNormalization()(hidden)
            hidden = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(hidden)
            hidden = BatchNormalization()(hidden)
            output = Dense(978, activation='tanh', kernel_regularizer=l2(0.01))(hidden)
            output = Lambda(lambda x:10*x)(output)
            
            model = Model(inputs=input_drug, outputs=output)
        
        elif (self.property_only):
            input_drug = Input(shape=(self.property_dim,))

            hidden = Dense(128, input_dim=self.property_dim, activation='relu', kernel_regularizer=l2(0.01))(input_drug)
            hidden = BatchNormalization()(hidden)
            output = Dense(978, activation='tanh', kernel_regularizer=l2(0.01))(hidden)
            output = Lambda(lambda x:10*x)(output)
            
            model = Model(inputs=input_drug, outputs=output)
            
        else:
            structure = Input(shape=(self.structure_dim,))

            hidden = Dense(512, input_dim=self.structure_dim, activation='relu', kernel_regularizer=l2(0.01))(structure)
            hidden = BatchNormalization()(hidden)
            hidden = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(hidden)
            hidden = BatchNormalization()(hidden)

            prop = Input(shape=(self.property_dim,))

            hidden2 = Dense(128, input_dim=self.property_dim, activation='relu', kernel_regularizer=l2(0.01))(prop)
            hidden2 = BatchNormalization()(hidden2)

            concat = concatenate([hidden, hidden2])
            output = Dense(978, activation='tanh', kernel_regularizer=l2(0.01))(concat)

            output = Lambda(lambda x:10*x)(output)
            
            model = Model(inputs=[structure, prop], outputs=output)

        model.compile(loss="mean_squared_error", optimizer=Adam(0.0001), metrics=[metrics.mse, self.tf_pearson])
        
        return model
        
    def get_model_summary(self):
        return self.model.summary()
    
    def tf_pearson(self, y_true, y_pred):
        return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]
    
    def set_checkpoint(self):
        checkpoint = ModelCheckpoint(self.model_save_path+self.model_name+'.h5', monitor='loss', verbose=1, save_best_only=self.save_best_only, mode='min', period=1)
        self.callbacks.append(checkpoint)
        self.callbacks.append(CosineAnnealingScheduler(T_max=self.t_max, eta_max=self.eta_max))
                
    def train(self, train_x, train_y, model_save_path, model_name, validation_split=0.1, batch_size=64, epochs=1000, cosineAnnealing_tmax=20, eta_max=1e-4, save_best_only=True):
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.t_max = cosineAnnealing_tmax
        self.eta_max = eta_max
        self.save_best_only = save_best_only
        
        self.callbacks = []
        self.set_checkpoint()
        
        self.model.fit(train_x, train_y, validation_split=validation_split, shuffle=True, verbose=0, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks)
        self.history = self.model.history
    
    def test(self, test_x, test_y):
        recon_x = self.model.predict(test_x)
        
        corr_list = list()
        for i in range(recon_x.shape[0]):
            corr_list.append(np.corrcoef(test_y.values[i], recon_x[i])[0][1])
        print('Average Correlation: ', np.mean(corr_list))
        
        return corr_list
    
    def predict(self, x):
        return self.model.predict(x)
        
    def save_model(self, model_save_path, model_name):
        model_json = self.model.to_json()
        with open(model_save_path+model_name+'.json', 'w') as json_file:
            json_file.write(model_json)

        self.model.save_weights(model_save_path+model_name+'_weights.h5')
        
    def load_model(self, model_load_path, model_weights):
#         custom_objects = {'tf_pearson':self.tf_pearson}
        self.model.load_weights(model_load_path+model_weights)


class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)