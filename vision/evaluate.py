"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

import model.input_fn
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/training_one_example',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='last_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_data_dir = args.data_dir

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]


    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = model.input_fn.load_data('data/eval_images')

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', params, reuse=False)
    print(type(test_inputs))
    logging.info("Starting evaluation")
    evaluate(model_spec, test_inputs, args.model_dir, params, args.restore_from)
