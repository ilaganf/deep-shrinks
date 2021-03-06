"""Train the model"""
import matplotlib
matplotlib.use('TkAgg')
import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import random

import tensorflow as tf

import model.input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/training_one_example',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    # logging.info("Creating the datasets...")
    # data_dir = args.data_dir
    # train_data_dir = data_dir
    # eval_data_dir = data_dir ###CHANGE BACK
    # # dev_data_dir = os.path.join(data_dir, "dev_signs")
    #
    # # Get the filenames from the train and dev sets
    # train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
    #                    if f.endswith('.jpg')]
    # eval_filenames = [os.path.join(eval_data_dir, f) for f in os.listdir(eval_data_dir)
    #                    if f.endswith('.jpg')]


    # Specify the sizes of the dataset we train on and evaluate on


    # Create the two iterators over the two datasets
    # train_inputs = input_fn(True, train_filenames, params)
    # eval_inputs = input_fn(False, eval_filenames, params)

    # Load image data
    train_data = model.input_fn.load_data('data/train_images')
    eval_data = model.input_fn.load_data('data/eval_images')

    params.train_size = len(train_data)
    params.eval_size = len(eval_data)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', params)
    eval_model_spec = model_fn('eval', params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_data, eval_data, train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
