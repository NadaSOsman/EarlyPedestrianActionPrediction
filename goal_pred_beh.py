from jaad_data import JAAD
import os
import sys
import yaml
import numpy as np
import getopt
import pickle
import cv2
import tensorflow as tf
import random as rn
from functools import partial
from itertools import product
from argparse import ArgumentParser

os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(1442)
rn.seed(12435)
from tensorflow.compat.v1.keras import backend as K

tf.random.set_seed(1234)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
#session_conf.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from tensorflow.keras.utils import Sequence, to_categorical, normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, Lambda, GlobalMaxPooling2D
from tensorflow.keras.layers import Activation, Concatenate, LayerNormalization, Conv2D, MaxPooling2D, Flatten, dot, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import vgg16, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import load_img



class GoalDataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=True,
                 global_pooling='max',
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False,
                 num_classes=1,
                 opts=None):

        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size  else batch_size
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()
        self.num_classes = num_classes
        self.opts = opts

    def get_size(self):
        return len(self.data[0])

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        X = self._generate_X(indices)
        Y = self._generate_y(indices)
        return X, Y

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            features_batch[i, ] = img_features
                        else:
                            features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        Y = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, self.data_sizes[input_type_idx][-1]))
            for i, index in enumerate(indices):
                if isinstance(self.labels[input_type_idx][index][0], str):
                    cached_path_list = self.labels[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            features_batch[i, ] = img_features
                        else:
                            features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.labels[input_type_idx][index][0]
            Y.append(features_batch)
        return Y



class GoalPrediction(object):
    def __init__(self):
        self._generator = None
        self._global_pooling = 'max'
        self._backbone = 'vgg16'

    def run(self, config_path, test):
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)

        print(configs['model_opts']['dataset'], '--------------------------------------')
        tte = configs['model_opts']['time_to_event']
        configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + 2*tte

        imdb = JAAD(data_path= configs['data_opts']['path_to_dataset'])

        data_raw_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        data_raw_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
        data_raw_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])

        data_train = self.get_data('train', data_raw_train, configs['model_opts'])
        test_data = self.get_data('test', data_raw_test, configs['model_opts'])
        val_data = self.get_data('val', data_raw_test, configs['model_opts'])

        print(data_train['count'])
        print(test_data['count'])
        model = self.lstm(opts=configs['model_opts'])
        if test:
            print("Testing...")
            model.load_weights(configs['model_opts']['model_path']+'/goal_beh'+'_'+
                               '_'.join(configs['model_opts']['obs_input_type'])+'_'+
                               str(configs['model_opts']['lr'])+'_'+
                               str(configs['model_opts']['hidden'])+'_'+
                               '.h5', by_name=False, skip_mismatch=False)

        else:
            optimizer = self.get_optimizer('adam')()
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

            model_path = configs['model_opts']['model_path']+'/goal_beh'+'_'+\
                             '_'.join(configs['model_opts']['obs_input_type'])+'_'+\
                             str(configs['model_opts']['lr'])+'_'+\
                             str(configs['model_opts']['hidden'])+'_'+'.h5'

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                                     save_weights_only=True,
                                                                     monitor='val_output_2_mean_squared_error',
                                                                     mode='min',
                                                                     save_best_only=True)

            history = model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=configs['model_opts']['batch_size'],
                                  epochs=configs['model_opts']['epochs'],
                                  validation_data=val_data['data'][0],
                                  verbose=1,
                                  callbacks=[checkpoint_callback])

            model.load_weights(model_path)

        print(model.evaluate(test_data['data'][0], verbose=1))


    def get_optimizer(self, optimizer):
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop


    def get_data(self, data_type, data_raw, model_opts):
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, labels, intent_count, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        _data = []
        data_sizes = []
        data_labels = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                label, label_shape = self.get_context_data(model_opts, labels, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose = model_opts['path_to_pose']
                features = self.get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                label = self.get_pose(labels['image'],
                                    labels['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                label = labels[d_type]
                feat_shape = features.shape[1:]

            _data.append(features)
            data_labels.append(label)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        if self._generator:
            _data = (GoalDataGenerator(data=_data,
                                   labels=data_labels,
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'] if data_type=='train' else 1,
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test',
                                   num_classes=model_opts['num_classes'],
                                   opts=model_opts), data_labels, data['lens'])
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count, 'intent_count':intent_count}}

    def get_data_sequence(self, data_type, data_raw, opts):
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        min_encoding_len = opts['min_encoding_len']
        num_classes = opts['num_classes']

        balance = opts['balance_data'] if data_type == 'train' else False
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])

        d['lens'] = d['box'].copy()
        all_goals = {}

        overlap = opts['overlap']
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res
        for k in d.keys():
            seqs = []
            lens = []
            goals = []
            for seq in d[k]:
                seq_len = len(seq)
                if(int(len(seq)%(time_to_event+min_encoding_len))!=0):
                    seq = [seq[0]]*(time_to_event+min_encoding_len-int(len(seq)%(time_to_event+min_encoding_len)))+seq

                seqs.extend([seq[len(seq)-obs_length-4*time_to_event:len(seq)-4*time_to_event]+seq[i:i+1] \
                            for i in range(len(seq)-4*time_to_event,len(seq))])
                lens.extend([min(seq_len, obs_length+4*time_to_event) for i in range(len(seq)-4*time_to_event,len(seq))])
                goals.extend([[seq[-1]] for i in range(len(seq)-4*time_to_event,len(seq))])
            if k == 'lens':
                d[k] = lens
            else:
                d[k] = seqs
                all_goals[k] = goals

        for k in d.keys():
            d[k] = np.array(d[k])

        labels = []
        crossing_count = 0
        intending_count = 0
        for l in d['crossing']:
            labels.append(l[0][0])
            if(l[0][0] == 1):
                crossing_count +=1
            if(l[0][0] == 2):
                intending_count +=1

        del data_raw
        notcrossing_count = len(d['crossing']) - crossing_count - intending_count
        return d, all_goals, intending_count, notcrossing_count, crossing_count


    def get_context_data(self, model_opts, data, data_type, feature_type):
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'scene_context' in feature_type:
            data_gen_params['crop_type'] = 'none'
        save_folder_name = feature_type
        save_folder_name = '_'.join([feature_type, aux_name])
        if 'local_context' in feature_type or 'surround' in feature_type:
            save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'],_ = self.get_path(save_folder=save_folder_name,
                                         dataset=dataset, save_root_folder='data/features')
        return self.load_images_crop_and_process(data['image'],
                                                 data['box'],
                                                 data['ped_id'],
                                                 process=process,
                                                 **data_gen_params)


    def update_progress(self, progress):
        barLength = 20  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)

        block = int(round(barLength * progress))
        text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()


    def jitter_bbox(self, img_path, bbox, mode, ratio):
        assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
        'mode %s is invalid.' % mode

        if mode == 'same':
            return bbox

        img = load_img(img_path)

        if mode in ['random_enlarge', 'enlarge']:
            jitter_ratio = abs(ratio)
        else:
            jitter_ratio = ratio

        if mode == 'random_enlarge':
            jitter_ratio = np.random.random_sample() * jitter_ratio
        elif mode == 'random_move':
            # for ratio between (-jitter_ratio, jitter_ratio)
            # for sampling the formula is [a,b), b > a,
            # random_sample * (b-a) + a
            jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

        jit_boxes = []
        for b in bbox:
            bbox_width = b[2] - b[0]
            bbox_height = b[3] - b[1]

            width_change = bbox_width * jitter_ratio
            height_change = bbox_height * jitter_ratio

            if width_change < height_change:
                height_change = width_change
            else:
                width_change = height_change

            if mode in ['enlarge', 'random_enlarge']:
                b[0] = b[0] - width_change // 2
                b[1] = b[1] - height_change // 2
            else:
                b[0] = b[0] + width_change // 2
                b[1] = b[1] + height_change // 2

            b[2] = b[2] + width_change // 2
            b[3] = b[3] + height_change // 2

            # Checks to make sure the bbox is not exiting the image boundaries
            b = self.bbox_sanity_check(img.size, b)
            jit_boxes.append(b)
        # elif crop_opts['mode'] == 'border_only':
        return jit_boxes

    def bbox_sanity_check(self, img_size, bbox):
        img_width, img_heigth = img_size
        if bbox[0] < 0:
            bbox[0] = 0.0
        if bbox[1] < 0:
            bbox[1] = 0.0
        if bbox[2] >= img_width:
            bbox[2] = img_width - 1
        if bbox[3] >= img_heigth:
            bbox[3] = img_heigth - 1
        return bbox


    def img_pad(self, img, mode='warp', size=224):
        assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
        image = np.copy(img)
        if mode == 'warp':
            warped_image = cv2.resize(img, (size, size))
            return warped_image
        elif mode == 'same':
            return image
        elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
            img_size = image.shape[:2][::-1] # original size is in (height, width)
            ratio = float(size)/max(img_size)
            if mode == 'pad_resize' or \
              (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
                img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
                image = cv2.resize(image, img_size)
            padded_image = np.zeros((size, size)+(image.shape[-1],), dtype=img.dtype)
            w_off = (size-img_size[0])//2
            h_off = (size-img_size[1])//2
            padded_image[h_off:h_off + img_size[1], w_off:w_off+ img_size[0],:] = image
            return padded_image


    def squarify(self, bbox, squarify_ratio, img_width):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * squarify_ratio - width
        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2
        # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox


    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=True,
                                     regen_data=False):

        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input, 'mobilenet': mobilenet.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'mobilenet': mobilenet.MobileNet}
        print("Backbone Models Loaded ......")

        preprocess_input = preprocess_dict.get(self._backbone, None)
        print("Preprocessing Model:", self._backbone)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50', 'mobilenet']), "{} is not supported".format(self._backbone)

        print("Initializing Preprocessin Model.......")
        convnet =  backbone_dict[self._backbone](input_shape=(224, 224, 3), include_top=False, weights='./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        print("Preprocessing Model Initialized........")
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            self.update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(imp)
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                    else:
                        img_data = cv2.imread(imp)
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            b = list(map(int, b[0:4]))
                            cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                            img_features = self.img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = self.jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = self.squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = self.img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = self.jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = self.squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = self.img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape


    def get_path(self, file_name='',
             sub_folder='',
             save_folder='models',
             dataset='jaad',
             save_root_folder='data/'):

        save_path = os.path.join(save_root_folder, dataset, save_folder, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path


    def flip_pose(self, pose):
        flip_map = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25,
                    26, 27, 16, 17, 18, 19, 20, 21, 30, 31, 28, 29, 34, 35, 32, 33]
        new_pose = pose.copy()
        flip_pose = [0] * len(new_pose)
        for i in range(len(new_pose)):
            if i % 2 == 0 and new_pose[i] != 0:
                new_pose[i] = 1 - new_pose[i]
            flip_pose[flip_map[i]] = new_pose[i]
        return flip_pose


    def get_pose(self, img_sequences, ped_ids, file_path, data_type='train', dataset='jaad'):
        poses_all = []
        set_poses_list = [x for x in os.listdir(file_path) if x.endswith('.pkl')]
        set_poses = {}
        for s in set_poses_list:
            with open(os.path.join(file_path, s), 'rb') as fid:
                try:
                    p = pickle.load(fid)
                except:
                    p = pickle.load(fid, encoding='bytes')
            set_poses[s.split('.pkl')[0].split('_')[-1]] = p
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            #update_progress(i / len(img_sequences))
            pose = []
            for imp, p in zip(seq, pid):
                flip_image = False

                if dataset == 'pie':
                    set_id = imp.split('/')[-3]
                elif dataset == 'jaad':
                    set_id = 'set01'

                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                if 'flip' in img_name:
                    img_name = img_name.replace('_flip', '')
                    flip_image = True
                k = img_name + '_' + p[0]
                """if(vid_id not in set_poses[set_id]):
                    if(len(pose) != 0):
                        pose.append(pose[-1])
                    else:
                        pose.append([0] * 36)"""
                if k in set_poses[set_id][vid_id].keys():
                    # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne,
                    #  Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
                    if flip_image:
                        pose.append(self.flip_pose(set_poses[set_id][vid_id][k]))
                    else:
                        pose.append(set_poses[set_id][vid_id][k])
                else:
                    pose.append([0] * 36)
            #print(pose)
            poses_all.append(pose)
        poses_all = np.array(poses_all)
        return poses_all

    def lstm(self, opts=None):
        inputs = []
        encodings = []
        current_features = []
        lstms = []
        outputs = []

        for i in range(len(opts['obs_input_type'])):
            inputs.append(Input(shape=(17, int(opts['feat_size'][i]))))
            encodings.append(Lambda(lambda x: x[:,:16])(inputs[i]))
            current_features.append(Lambda(lambda x: x[:,-1])(inputs[i]))

        lstms.append(LSTM(int(opts['feat_size'][0]), dropout=0.2, return_sequences=True)(encodings[0]))
        outputs.append(Dense(int(opts['feat_size'][0]), name='output_0')(Concatenate()([Lambda(lambda x: x[:,-1])(lstms[0]), current_features[0]])))
        for i in range(1,len(inputs)):
            lstms.append(LSTM(int(opts['feat_size'][i]), dropout=0.2, return_sequences=True)(Concatenate(axis=-1)([lstms[i-1],encodings[i]])))
            outputs.append(Dense(int(opts['feat_size'][i]), name='output_'+str(i))(Concatenate()([Lambda(lambda x: x[:,-1])(lstms[i]), current_features[i]])))

        model = Model(inputs, outputs, name='lstm')
        print(model.summary())
        return model


if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for Goal_Prediction")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    goal_prediction = GoalPrediction()
    goal_prediction.run(args.config_file, args.test)
