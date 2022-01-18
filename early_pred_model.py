import os
import sys
import numpy as np
import tensorflow as tf
from functools import partial
from itertools import product
from tensorflow.keras.utils import Sequence, to_categorical, normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, Lambda, GlobalMaxPooling2D, Concatenate, BatchNormalization, Softmax
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from early_pred_data_generator import DataGenerator, DataLoader
from tensorflow.compat.v1.keras import backend as K


class EarlyPredictionModel(object):
    def __init__(self, opts=None, data_generators={}, pretrain=False, fusion=False):
        self.opts = opts
        self.data_generators = data_generators
        self.pretrain = pretrain
        self.fusion = fusion

        if pretrain:
            model_name = "/pretrained_"+opts['model_opts']['dataset']+"_"+opts['data_opts']['sample_type']
        elif fusion:
            model_name = "/fusion_"+opts['model_opts']['dataset']+"_"+opts['data_opts']['sample_type']
        else:
            model_name = "/trained_"+opts['model_opts']['dataset']+"_"+opts['data_opts']['sample_type']

        self.model_path = opts['model_opts']['model_path']+model_name+'_'+\
                          '_'.join(opts['model_opts']['obs_input_type'])+'_'+\
                          str(opts['model_opts']['lr'])+'_'+\
                          str(opts['model_opts']['hidden'])+'_'+'.h5'

        self.model = None if self.fusion else self.rulstm()
        if fusion:
            print("Loading Trained Model for Fusion...")
            models = []
            model_name = "/trained_"+self.opts['model_opts']['dataset']+"_"+self.opts['data_opts']['sample_type']
            for i in range(len(self.opts['model_opts']['obs_input_type'])):
                modality_model = self.rulstm(feat_size=self.opts['model_opts']['feat_size'][i])
                modality_model.load_weights(self.opts['model_opts']['model_path']+model_name+'_'+
                               self.opts['model_opts']['obs_input_type'][i]+'_'+
                               str(self.opts['model_opts']['lr'])+'_'+
                               str(self.opts['model_opts']['hidden'])+'_'+
                               '.h5', by_name=False, skip_mismatch=False)
                models.append(modality_model)
            self.model = self.fusion_rusltm(models)
        elif not pretrain:
            print("Loading Pretrained...")
            model_name = "/pretrained_"+self.opts['model_opts']['dataset']+"_"+self.opts['data_opts']['sample_type']
            self.model.load_weights(self.opts['model_opts']['model_path']+model_name+'_'+
                               '_'.join(self.opts['model_opts']['obs_input_type'])+'_'+
                               str(self.opts['model_opts']['lr'])+'_'+
                               str(self.opts['model_opts']['hidden'])+'_'+
                               '.h5', by_name=False, skip_mismatch=False)


    def train(self):
        class_w = self.class_weights(self.opts['model_opts']['apply_class_weights'], self.data_generators['train']['count'])
        optimizer = self.get_optimizer(self.opts['model_opts']['optimizer'])()
        if(self.opts['model_opts']['apply_class_weights']):
            w = [class_w[0], class_w[1]]
            self.model.compile(loss=self.weighted_binary_crossentropy(weights=w), optimizer=optimizer, metrics=['accuracy'])
        else:
            self.model.compile(loss=self.opts['model_opts']['classifier_loss'], optimizer=optimizer, metrics=['accuracy'])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                 save_weights_only=True,
                                                                 monitor='val_tf_op_layer_Sum_30_accuracy' if self.fusion else 'val_output_36_accuracy',
                                                                 mode='max',
                                                                 save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        history = self.model.fit(x=self.data_generators['train']['data'][0],
                                 y=None if self.data_generators['train']['data'][0] else self.data_generators['train']['data'][1],
                                 batch_size=self.opts['model_opts']['batch_size'],
                                 epochs=self.opts['model_opts']['epochs'],
                                 validation_data=self.data_generators['val']['data'][0],
                                 verbose=1,
                                 callbacks=[tensorboard_callback, checkpoint_callback])



    def test(self):
        self.model.load_weights(self.model_path)
        test_results = self.model.predict(self.data_generators['test']['data'][0], verbose=1)
        test_results_array = np.array(np.round(test_results))
        average = 'binary'
        multi_class = 'raise'

        AT = np.flip(np.arange(0, 4.1, 0.1))
        count = 0
        index = int(configs['model_opts']['interval']/configs['model_opts']['step'])
        masking_index = (test_data['data'][2]/configs['model_opts']['step']).astype(int)
        for i in range(len(test_results)):
            rev_index = int((configs['model_opts']['seq_len']-configs['model_opts']['obs_length'])/configs['model_opts']['step'])\
                        + int(configs['model_opts']['obs_length']/configs['model_opts']['step']) - i
            acc = self.accuracy(test_data['data'][1][i], test_results_array[i], rev_index, masking_index)
            f1 = self.f1(test_data['data'][1][i], test_results_array[i], rev_index, masking_index, average=average)
            auc = self.auc(test_data['data'][1][i], test_results_array[i], rev_index, masking_index, multi_class=multi_class)
            precision = self.precision(test_data['data'][1][i], test_results_array[i], rev_index, masking_index, average=average)
            recall = self.recall(test_data['data'][1][i], test_results_array[i], rev_index, masking_index, average=average)

            print(AT[count],':' ,'acc:', acc, '- auc:', auc, '- f1:', f1, '- precision:', precision, '- recall:', recall)
            count += 1

    def accuracy(self, true, pred, index, masking_index):
        masking_index = masking_index >= index
        y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
        y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
        return accuracy_score(y_true, y_pred)

    def f1(self, true, pred, index, masking_index, average):
        masking_index = masking_index >= index
        y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
        y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
        return f1_score(y_true, y_pred, average=average)

    def auc(self, true, pred, index, masking_index, multi_class):
        masking_index = masking_index >= index
        y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
        y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
        return roc_auc_score(y_true, y_pred, multi_class=multi_class)

    def precision(self, true, pred, index, masking_index, average):
        masking_index = masking_index >= index
        y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
        y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
        return precision_score(y_true, y_pred, average=average)


    def recall(self, true, pred, index, masking_index, average):
        masking_index = masking_index >= index
        y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
        y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
        return recall_score(y_true, y_pred, average=average)


    def class_weights(self, apply_weights, sample_count):
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']
        # formula from sklearn
        #neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        #pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0

        # use simple ratio
        neg_weight = sample_count['pos_count']/total
        pos_weight = sample_count['neg_count']/total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}


    def get_optimizer(self, optimizer):
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def weighted_binary_crossentropy(self, weights):
        def loss_func(y_true, y_pred):
            tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
            tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
            weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
            ce = K.binary_crossentropy(y_pred, y_true)
            loss = K.mean(tf.multiply(ce, weights_v))
            return loss
        return loss_func

    def fusion_rusltm(self, models=[]):
        step = self.opts['step']
        units = 2*len(models)*self.opts['hidden']
        inputs = []
        models_pred = []
        models_contexts = []
        seq_len = int((self.opts['seq_len']-self.opts['obs_length'])/step)
        for model in models:
            for layer in model.layers:
                layer._name = layer.name + str("__") + str(len(inputs))
            inputs.append(model.inputs)
            models_pred.append(tf.convert_to_tensor(model.outputs[:seq_len]))
            models_contexts.append(tf.convert_to_tensor(model.outputs[seq_len:]))

        context = tf.concat(models_contexts, -1)
        preds = tf.concat(models_pred, -1)

        d1 = Dense(int(units/4), activation='relu')
        dr1 = Dropout(opts['dropout'])
        d2 = Dense(int(units/8), activation='relu')
        dr2 = Dropout(opts['dropout'])
        d3 = Dense(len(models), activation='relu')
        s = Softmax()

        outputs = []
        for i in range(context.shape[0]):
            c = Lambda(lambda s, i=i: s[i])(context)
            p = Lambda(lambda s, i=i: s[i])(preds)
            x = d1(c)
            x = dr1(x)
            x = d2(x)
            x = dr2(x)
            x = d3(x)
            x = s(x)

            weighted_preds = p * x
            weighted_preds = tf.reduce_sum(weighted_preds, axis=-1, keepdims=True)
            outputs.append(weighted_preds)

        model = Model(inputs=inputs, outputs=outputs, name='fusion')
        return model

    def rulstm(self, feat_size=40):
        input_seq = Input((self.opts['model_opts']['seq_len'], feat_size if self.fusion else self.opts['model_opts']['feat_size']))
        step = self.opts['model_opts']['step']
        sampled_seq = tf.concat([Lambda(lambda s: s[:,0:self.opts['model_opts']['obs_length']:step])(input_seq),\
                                 Lambda(lambda s: s[:,self.opts['model_opts']['obs_length']::step])(input_seq)], 1)
        sampled_seq = BatchNormalization()(sampled_seq)
        seq_len = sampled_seq.shape[1]
        outputs = []
        contexts = []
        h = None
        c = None
        rolling_lstm = LSTM(self.opts['model_opts']['hidden'], dropout=self.opts['model_opts']['dropout'], return_state=True)
        unrolling_lstm = LSTM(self.opts['model_opts']['hidden'], dropout=self.opts['model_opts']['dropout'])
        for i in range(seq_len):
            input = Lambda(lambda s,i=i: s[:,i:i+1])(sampled_seq)
            if(h != None and c != None):
                _, h, c = rolling_lstm(input, initial_state=[h, c])
            else:
                _, h, c = rolling_lstm(input)

            if self.pretrain:
                input_unrolling = Lambda(lambda s,i=i: s[:,i:])(sampled_seq)
            else:
                input_unrolling = RepeatVector(seq_len-i)(Lambda(lambda s,i=i: s[:,i])(sampled_seq))
            x_h = unrolling_lstm(input_unrolling, initial_state=[h, c])
            x = Dropout(self.opts['model_opts']['dropout'])(x_h)
            x = Dense(int(self.opts['model_opts']['hidden']/2), activation='relu')(x)
            x = Dense(int(self.opts['model_opts']['hidden']/4), activation='relu')(x)
            x = Dense(int(self.opts['model_opts']['hidden']/8), activation='relu')(x)
            y = Dense(self.opts['model_opts']['num_classes'], activation=self.opts['model_opts']['classifier_activation'], name='output_'+str(i))(x)
            if(i>int(self.opts['model_opts']['obs_length']/step)):
                outputs.append(y)
                contexts.append(tf.concat([x_h, c], -1))

        model_outputs = outputs+contexts if self.fusion else outputs
        model = Model(input_seq, model_outputs, name='rulstm')
        return model
