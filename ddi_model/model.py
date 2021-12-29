import keras
from keras.layers import *
from keras.regularizers import *
import keras.backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam

from DeSIDE_DDI_functions import *

class DDI_model(object):
    def __init__(self,input_drug_dim=978, input_se_dim=1, drug_emb_dim=100, se_emb_dim=100, output_dim=1, margin=1, drug_activation='elu'):
        
        self.input_drug_dim = input_drug_dim
        self.input_se_dim = input_se_dim
        self.drug_emb_dim = drug_emb_dim
        self.se_emb_dim = se_emb_dim
        self.output_dim = output_dim
        self.margin = margin
        self.drug_activation = drug_activation

        self.callbacks = []
        self.model = self.build()
        
    def build(self):
        drug1_exp = Input(shape=(self.input_drug_dim, ))
        drug2_exp = Input(shape=(self.input_drug_dim, ))
        
        shared_layer = Sequential(name='drug_embed_shared')
        shared_layer.add(Dense(output_dim=self.input_drug_dim, activation=self.drug_activation))
        shared_layer.add(BatchNormalization())

        drug1 = shared_layer(drug1_exp)
        drug2 = shared_layer(drug2_exp)
        
        concat = Concatenate()([drug1, drug2])
        
        glu1 = Dense(self.input_drug_dim, activation='sigmoid', name='drug1_glu')(concat)
        glu2 = Dense(self.input_drug_dim, activation='sigmoid', name='drug2_glu')(concat)

        drug1_selected = Multiply()([drug1, glu1])
        drug2_selected = Multiply()([drug2, glu2])
        drug1_selected = BatchNormalization()(drug1_selected)
        drug2_selected = BatchNormalization()(drug2_selected)
        
        shared_layer2 = Sequential(name='drug_embed_shared2')
        shared_layer2.add(Dense(output_dim=self.drug_emb_dim, kernel_regularizer=l2(0.001), activation=self.drug_activation))
        shared_layer2.add(BatchNormalization())

        drug1_emb = shared_layer2(drug1_selected)
        drug2_emb = shared_layer2(drug2_selected)
        
        # side effect
        input_se = Input(shape=(self.input_se_dim,))
        se_emb = Embedding(963, output_dim=self.se_emb_dim, input_length=self.input_se_dim)(input_se)

        # one-hot side effect for metric
        input_se_one_hot = Input(shape=(963,))

        # side effect mapping matrix
        se_head = Embedding(963, output_dim=self.drug_emb_dim*self.se_emb_dim, input_length=self.input_se_dim, embeddings_regularizer=l2(0.01))(input_se)
        se_head = Reshape((self.se_emb_dim, self.drug_emb_dim))(se_head)
        se_tail = Embedding(963, output_dim=self.drug_emb_dim*self.se_emb_dim, input_length=self.input_se_dim, embeddings_regularizer=l2(0.01))(input_se)
        se_tail = Reshape((self.se_emb_dim, self.drug_emb_dim))(se_tail)
        
        # MhH & MtT
        mh_dx = Dot(axes=(2,1))([se_head, drug1_emb])
        mt_dy = Dot(axes=(2,1))([se_tail, drug2_emb])
        mh_dy = Dot(axes=(2,1))([se_head, drug2_emb])
        mt_dx = Dot(axes=(2,1))([se_tail, drug1_emb])

        # || MhH + r - MtT ||
        score1 = add([mh_dx, se_emb])
        score1 = subtract([score1, mt_dy])
        score1 = Lambda(lambda x:K.sqrt(K.sum(K.square(x), axis=-1)))(score1)
        score1 = Reshape((1,))(score1)

        score2 = add([mh_dy, se_emb])
        score2 = subtract([score2, mt_dx])
        score2 = Lambda(lambda x:K.sqrt(K.sum(K.square(x), axis=-1)))(score2)
        score2 = Reshape((1,))(score2)

        final_score = add([score1, score2])

        model = Model(inputs=[drug1_exp, drug2_exp, input_se, input_se_one_hot], outputs=final_score)
        model.compile(loss=self.custom_loss_wrapper(se_one_hot=input_se_one_hot,margin=self.margin), \
                                 optimizer=Adam(0.001), metrics=['accuracy'])
        
        return model
    
    def custom_loss_wrapper(self, se_one_hot, margin):
        def custom_margin_loss(y_true, y_pred, se_one_hot=se_one_hot, margin=margin):
            pos_score = (y_true*y_pred)
            neg_score = (K.abs(K.ones_like(y_true)-y_true)*y_pred)

            se_pos = K.dot(K.transpose(pos_score), se_one_hot)
            se_neg = K.dot(K.transpose(neg_score), se_one_hot)

            se_mask = K.cast(se_pos*se_neg, dtype=bool)

            se_pos_score = K.cast(se_mask, dtype='float32')*se_pos
            se_neg_score = K.cast(se_mask, dtype='float32')*se_neg

            score = se_pos_score-se_neg_score+(K.ones_like(se_pos_score)*K.cast(se_mask, dtype='float32'))*margin
            final_loss = K.sum(K.maximum(K.zeros_like(score),score))

            return final_loss
        return custom_margin_loss
        
    def get_model_summary(self):
        return self.model.summary()
    
    def set_checkpoint(self):
        checkpoint= CustomModelCheckPoint(save_path=self.model_save_path, model_name=self.model_name, \
                                      init_learining_rate=self.init_lr, decay_rate=self.decay_rate, decay_steps=self.decay_steps)
        self.callbacks.append(checkpoint)
    
    def train(self, train_data, exp_df, split_frac, sampling_size, model_save_path, model_name, init_lr=0.0001, decay_rate=0.9, decay_steps=2, batch_size=1024):
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        
        self.callbacks = []
        self.set_checkpoint()
        
        optimal_threshold = pd.DataFrame(np.array(range(0,len(train_data.SE.unique()))), columns=['SE'])

        for n in range(sampling_size):
            print(n, ' Sample =======')
            cv_test = train_data.groupby(['SE', 'label']).apply(pd.DataFrame.sample, frac=split_frac)
            cv_test_x = cv_test.reset_index(drop=True).iloc[:,:3]
            cv_test_y = cv_test.reset_index(drop=True).iloc[:,-1]

            cv_train_data_rest = pd.concat([train_data, cv_test]).drop_duplicates(keep=False, inplace=False)
            cv_train_x = cv_train_data_rest.iloc[:,:3]
            cv_train_y = cv_train_data_rest.iloc[:,3]
            print('Cross validation train, test dataset size: ', cv_train_x.shape, cv_test_x.shape)

            cv_train_gen = custom_dataGenerator(cv_train_x, cv_train_y.values, batch_size=self.batch_size, exp_df=exp_df)
            cv_test_gen = custom_dataGenerator(cv_test_x, cv_test_y.values, batch_size=self.batch_size, exp_df=exp_df, shuffle=False) 

            steps_per_epoch = cv_train_x.shape[0] // self.batch_size // 10

            #======================================================================================================================#
            self.model.fit_generator(generator=cv_train_gen, steps_per_epoch=steps_per_epoch, validation_data=cv_test_gen, \
                                                       epochs=10, verbose=0, shuffle=True, callbacks=self.callbacks)

            cv_test_pred_y = self.model.predict_generator(generator=cv_test_gen)

            cv_test_prediction_scores = mean_predicted_score(cv_test, cv_test_pred_y, with_plot=False)
            cv_test_prediction_perf = cal_performance(cv_test_prediction_scores)
            print('AUC: {:.3f}, AUPR: {:.3f}'.format(cv_test_prediction_perf.describe().loc['mean']['AUC'], cv_test_prediction_perf.describe().loc['mean']['AUPR']))

            optimal_threshold = pd.concat([optimal_threshold, pd.DataFrame(cv_test_prediction_perf.optimal_thr).reset_index(drop=True)], axis=1)

        self.optimal_threshold = optimal_threshold
        self.history = self.model.history
        
    def test(self, test_x, test_y, exp_df):
        switch_x = test_x[['drug2','drug1','SE']]
        switch_x.columns = ['drug1','drug2','SE']
        
        ori_test_prediction_predicted_label_df, ori_test_prediction_perf_df, thr = external_validation_v2(self.model, test_x, test_y, exp_df=exp_df, optimal_threshold=self.optimal_threshold, batch_size=self.batch_size)
        swi_test_prediction_predicted_label_df, swi_test_prediction_perf_df, thr = external_validation_v2(self.model, switch_x, test_y, exp_df=exp_df, optimal_threshold=self.optimal_threshold, batch_size=self.batch_size)
        print('Test set predicted === ')

        merge_predicted_label_df, merged_perf_df = merge_both_pairs(ori_test_prediction_predicted_label_df, swi_test_prediction_predicted_label_df, thr, 'optimal_thr')
        print('AUC: {:.3f}, AUPR: {:.3f}'.format(merged_perf_df.describe().loc['mean']['AUC'], merged_perf_df.describe().loc['mean']['AUPR']))
        return merge_predicted_label_df, merged_perf_df
        
    def save_model(self):
        self.model.save(self.model_save_path+'final_{}.h5'.format(self.model_name))
        print('Model saved === ')
    
    def load_model(self, model_load_path, model_name, threshold_name):
        self.model.load_weights(model_load_path+model_name)
        self.optimal_threshold = pd.read_csv(model_load_path+threshold_name, index_col=0)
        
    def predict(self, x, exp_df):
        y = np.zeros(x.shape[0])
        
        test_gen = custom_dataGenerator(x, y, batch_size=self.batch_size, exp_df=exp_df, shuffle=False)
        pred_y = self.model.predict_generator(generator=test_gen)
        predicted_result = mean_predicted_score(pd.concat([x, pd.DataFrame(y, columns=['label'])], axis=1), pred_y, with_plot=False)
        predicted_label, thr = calculate_predicted_label_ver3(predicted_result, self.optimal_threshold)
        predicted_label = predicted_label[['drug1','drug2','SE','predicted_label','predicted_score']]
        
        return predicted_label