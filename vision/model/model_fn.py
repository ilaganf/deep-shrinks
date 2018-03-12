"""Define the model."""

import tensorflow as tf
import numpy as np

'''
ARCHITECTURE (Jiang et al.)
c = number of channels (3 for RGB)
ComCNN (Cr): 3 weight layers -    CONV -> ReLU       ->     CONV -> ReLU              ->     CONV
                                (64 3x3xc filters)     (stride 2, 64 3x3x64 filters)  (c 3x3x64 filters)

RecCNN (Re): 20 layers -    CONV->ReLU       -> CONV->BatchNorm->ReLU (x18)   ->       CONV
                         (64 3x3xc filters)      (64 3x3x64 filters)           (c 3x3x64 filters)

All convolutional layers use same padding
'''

def comCNN(inputs, params, num_channels, num_filters):
    '''Builds the ComCNN

        Defines the compressor convolutional neural network
    '''
    out = inputs
    with tf.variable_scope('ComCNN_vars'):
        with tf.variable_scope('ComCNN_block_1'):
            # TODO: possibly add regularization
            out = tf.layers.conv2d(inputs=out, filters=num_filters, 
                                   kernel_size=3, strides=1, padding='same')
            out = tf.nn.relu(out)
        with tf.variable_scope('ComCNN_block_2'):
            out = tf.layers.conv2d(out, num_filters, 3, 2, padding='same')
            out = tf.nn.relu(out)
        with tf.variable_scope('ComCNN_output_block'): 
            # for generating compact representation of image
            out = tf.layers.conv2d(out, num_channels, 3, 1, padding='same')

    return out


def recCNN(inputs, params, num_channels, num_filters, is_training):
    '''Builds the RecCNN

    Responsible for defining the reconstructor convolutional neural network
    '''
    out = inputs
    with tf.variable_scope('RecCNN_vars'):
        bn_momentum = params.bn_momentum
        up_size = params.image_size
        num_intermediate = 18
        out = tf.image.resize_images(out, (up_size, up_size), method=tf.image.ResizeMethod.BICUBIC) # Paper uses bicubic interpolation
        with tf.variable_scope('RecCNN_block_1'):
            out = tf.layers.conv2d(out, num_filters, 3, 1, padding='same')
            out = tf.nn.relu(out)
        for i in range(num_intermediate):
            with tf.variable_scope('RecCNN_block_{}'.format(i+2)):
                out = tf.layers.conv2d(out, num_filters, 3, 1, padding='same', activation=None)
                # out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
                out = tf.nn.relu(out)
        with tf.variable_scope('RecCNN_output_block'):
            out = tf.layers.conv2d(out, num_channels, 3, 1, padding='same')

    return out


def get_rec_input(compact, params):
    '''Creates the operation that calculates the input to the recCNN

    Takes the compact representation from the comCNN, encodes it as jpeg,
    then upscales to the pre-defined size using Bicubic interpolation
    '''
    compact = tf.image.convert_image_dtype(compact, tf.uint8, saturate=True)
    encode_decode = []
    encoded = tf.map_fn(lambda x: tf.image.encode_jpeg(x), compact, dtype=tf.string)
    decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x), encoded, dtype=tf.uint8)
    up = tf.image.resize_images(decoded, (params.image_size, params.image_size),
                                method=tf.image.ResizeMethod.BICUBIC)
    rec_input = tf.cast(up, tf.float32)
    return rec_input


def model_fn(mode, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    num_channels = params.num_channels
    num_filters = 64
    # labels = inputs['images'] # we train based on similarity to original image
    labels = tf.placeholder(tf.float32, shape=[None, params.image_size, params.image_size, num_channels], 
                            name='input_images')
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output of the model
        com = comCNN(labels, params, num_channels, num_filters)
        compress = get_rec_input(com, params) # puts compact representation through codec and upsizes
        x_hat = tf.placeholder(tf.float32, shape=[None, params.image_size, params.image_size, num_channels],
                               name="x_hat")
        rec = recCNN(x_hat, params, num_channels, num_filters, is_training)
        rec_output = tf.placeholder(tf.float32, shape=[None, params.image_size, params.image_size, num_channels],
                                    name="rec_output")
        com_direct = tf.image.resize_images(com, (params.image_size, params.image_size),
                                          method=tf.image.ResizeMethod.BICUBIC)
    
    final_output = rec_output + com_direct
    # Define loss for both networks
    com_loss = .5 * tf.losses.mean_squared_error(labels=labels, predictions=final_output)
    # com_loss = .5 * tf.losses.mean_squared_error(labels=labels, predictions=com_temp)
    rec_loss = .5 * tf.losses.mean_squared_error(labels=x_hat-labels, predictions=rec)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        # if params.use_batch_norm:
        #     # Add a dependency to update the moving mean and variance for batch normalization
        #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #         train_op = optimizer.minimize(loss, global_step=global_step)
        # else:
        train_op1 = optimizer.minimize(com_loss, global_step=global_step, 
                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                  scope='model/ComCNN_vars'))
        train_op2 = optimizer.minimize(rec_loss, global_step=global_step, 
                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                  scope='model/RecCNN_vars'))


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation
    # TODO: define metric for recCNN (MMSSIM)
    with tf.variable_scope("metrics"):
        metrics = {
            'com_loss': tf.metrics.mean(com_loss),
            'rec_loss': tf.metrics.mean(rec_loss),
            'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=compress+rec_output)
            # TODO: replace with MMSSIM
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('com_loss', com_loss)
    tf.summary.scalar('rec_loss', rec_loss)
    # tf.summary.scalar('MMSSIM', 0) TODO
    tf.summary.image('train_image', labels)

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    # mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    # for label in range(0, params.num_labels):
    #     mask_label = tf.logical_and(mask, tf.equal(predictions, label))
    #     incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
    #     tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = {}
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['compression_op'] = compress
    model_spec['reconstructed'] = rec
    model_spec["codec"] = tf.sigmoid(com)
    model_spec["final_output"] = tf.sigmoid(rec)
    model_spec['com_loss'] = com_loss
    model_spec['rec_loss'] = rec_loss
    if is_training:
        model_spec['com_train_op'] = train_op1
        model_spec['rec_train_op'] = train_op2
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['input'] = labels
    model_spec['x_hat_placeholder'] = x_hat
    model_spec['rec_placeholder'] = rec_output

    return model_spec
