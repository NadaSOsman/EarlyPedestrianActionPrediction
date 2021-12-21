import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import random as rn
from argparse import ArgumentParser

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1442)
rn.seed(12435)
from tensorflow.compat.v1.keras import backend as K

tf.random.set_seed(1234)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from early_pred_data_generator import DataGenerator, DataLoader
from early_pred_model import EarlyPrediction


if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for Early_Intention_Prediction")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fusion', action='store_true')

    args = parser.parse_args()

     with open((args.config_file, 'r') as f:
            configs = yaml.safe_load(f)

    data_loader = DataLoader(opts=configs, fusion=args.fusion)
    data_generators = dada_loader.get_data_generators(['train', 'test', 'val'])

    early_pred_model = EarlyPrediction(opts=args.config_file, data_generators=data_generators, pretrain=args.pretrain, fusion=args.fusion)

    if not args.test:
        early_pred_model.train()
    early_pred_model.test()


