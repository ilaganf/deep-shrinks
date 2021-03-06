"""Tensorflow utility functions for training"""

import logging
import os

from scipy import misc
from tqdm import trange
import tensorflow as tf
import numpy as np

from model.utils import save_dict_to_json
from model.evaluation import evaluate_sess
import matplotlib.pyplot as plt
from model.msssim import MultiScaleSSIM


def train_sess(data, sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training
    compress = model_spec['compression_op']
    com_loss = model_spec['com_loss']
    rec_loss = model_spec['rec_loss']
    com_train_op = model_spec['com_train_op']
    rec_train_op = model_spec['rec_train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    com = model_spec['codec']
    final_output = model_spec['final_output']
    rec = model_spec['reconstructed']
    global_step = tf.train.get_global_step()
    labels = model_spec['input']
    x_hat_feed = model_spec['x_hat_placeholder']
    rec_output = model_spec['rec_placeholder']
    # Load the training dataset into the pipeline and initialize the metrics local variables
    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])



    #Create optimizers for Com/Rec CNNs

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Get current minibatch
        batch_start = i * params.batch_size
        batch_end = (i+1) * params.batch_size
        batch = data[batch_start:batch_end]

        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            x_hat, global_step_val = sess.run([compress, global_step], feed_dict={labels:batch})
            _, rec_loss_val, residuals = sess.run([rec_train_op, rec_loss, rec],
                                                  feed_dict={x_hat_feed:x_hat, labels:batch})
            _, com_loss_val, com_output = sess.run([com_train_op, com_loss, com],
                                                   feed_dict={rec_output:residuals, labels:batch})

            summ, _ = sess.run([summary_op, update_metrics],
                               feed_dict={labels:batch, x_hat_feed:x_hat, rec_output:residuals})

            writer.add_summary(summ, global_step_val)
            plt.imshow(np.squeeze(com_output))
            plt.show()
        else:
            x_hat, global_step_val = sess.run([compress, global_step], feed_dict={labels:batch})
            _, rec_loss_val, residuals = sess.run([rec_train_op, rec_loss, rec],
                                                  feed_dict={x_hat_feed:x_hat, labels:batch})
            _, com_loss_val, com_output = sess.run([com_train_op, com_loss, com],
                                                   feed_dict={rec_output:residuals, labels:batch})
            # _, _, rec_loss_val = sess.run([rec_train_op, update_metrics, rec_loss])
        # Log the loss in the tqdm progress bar
        # t.set_postfix(loss='{:05.3f}'.format(com_loss_val))
        # t.set_postfix(loss='{:05.3f}'.format(rec_loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train: " + metrics_string)
    return batch, x_hat+residuals


def train_and_evaluate(train_data, eval_data, train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        # Reload weights from directory if specified
        if restore_from is not None:
            logging.info("Restoring parameters from {}".format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        best_eval_msssim = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            orig_train_img, result_train_img = train_sess(train_data, sess, train_model_spec, num_steps, train_writer, params)
            
            train_msssim = MultiScaleSSIM(orig_train_img, result_train_img)
            logging.info("Train MS-SSIM: {}".format(train_msssim))

            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)
            # Evaluate for one epoch on validation set
            if epoch % params.save_summary_steps == 0:
                num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
                metrics, orig_eval_img, result_eval_img = evaluate_sess(eval_data, sess, eval_model_spec, num_steps, eval_writer)
                eval_msssim = MultiScaleSSIM(orig_eval_img, result_eval_img)
                logging.info("Eval MS-SSIM: {}".format(eval_msssim))
                # If best_eval, best_save_path
                cur_eval_msssim = eval_msssim
                if cur_eval_msssim >= best_eval_msssim:
                    # Store new best accuracy
                    best_eval_msssim = cur_eval_msssim
                    # Save weights
                    best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                    best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                    logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                    # Save best eval metrics in a json file in the model directory
                    best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
                    save_dict_to_json(metrics, best_json_path)

                # Save latest eval metrics in a json file in the model directory
                last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
                save_dict_to_json(metrics, last_json_path)
