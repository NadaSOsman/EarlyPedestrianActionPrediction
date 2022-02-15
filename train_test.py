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
from early_pred_model import EarlyPredictionModel


if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for Early_Intention_Prediction")
    parser.add_argument('--configs_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--t_forcing', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--goal_based', action='store_true')
    parser.add_argument('--goal_conc', action='store_true')
    parser.add_argument('--goal_att', action='store_true')
    parser.add_argument('--goal_scheme', action='store_true')

    args = parser.parse_args()

    with open(args.configs_file, 'r') as f:
        configs = yaml.safe_load(f)

    data_loader = DataLoader(opts=configs, fusion=args.fusion)
    data_generators = data_loader.get_data_generators(['train', 'test', 'val'])

    early_pred_model = EarlyPredictionModel(opts=configs, data_generators=data_generators, pretrain=args.pretrain, fusion=args.fusion, goal_based=args.goal_based, goal_conc=args.goal_conc, goal_att=args.goal_att, goal_scheme=args.goal_scheme)

    if not args.test:
        early_pred_model.train()
    early_pred_model.test()


