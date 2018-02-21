"""Define the model."""

import tensorflow as tf


def comCNN(inputs, params, num_channels, num_filters):
    '''
    Builds the ComCNN
    '''
    out = inputs
    with tf.variable_scope('ComCNN_block_1'):
        out = tf.layers.conv2d(input=out, filter=[3, 3, num_channels, num_filters], strides=1, padding='same')
        out = tf.nn.relu(out)
    with tf.variable_scope('ComCNN_block_2'):
        out = tf.layers.conv2d(out, [3, 3, num_filters, num_filters], 2, padding='same')
        out = tf.nn.relu(out)
    with tf.variable_scope('ComCNN_output_block'): # for generating compact representation of image
        out = tf.layers.conv2d(out, [3, 3, num_filters, num_channels], 1, padding='same')

    return out

def recCNN(inputs, params, num_channels, num_filters):
    '''
    Builds the RecCNN
    '''
    out = inputs
    # RecCNN
    bn_momentum = params.bn_momentum
    up_size = params.image_size
    num_intermediate = 18
    out = tf.image.resize_images(out, up_size, method=ResizeMethod.BICUBIC) # Paper uses bicubic interpolation
    with tf.variable_scope('RecCNN_block_1'):
        out = tf.layers.conv2d(out, [3, 3, num_channels, num_filters], 1, padding='same')
        out = tf.nn.relu(out)
    for i in range(num_intermediate):
        with tf.variable_scope('RecCNN_block_{}'.format(i+2)):
            out = tf.layers.conv2d(out, [3, 3, num_filters, num_filters], 1, padding='same')
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
    with tf.variable_scope('RecCNN_output_block'):
        out = tf.layers.conv2d(out, [3, 3, num_filters, num_channels], 1, padding='same')

    return out


def build_model(is_training, inputs, params):
    """

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model

    ARCHITECTURE (Jiang et al.)
    c = number of channels (3 for RGB)
    ComCNN (Cr): 3 weight layers -   CONV -> ReLU       ->     CONV -> ReLU              ->     CONV
                                    (64 3x3xc filters)     (stride 2, 64 3x3x64 filters)  (c 3x3x64 filters)

    RecCNN (Re): 20 layers -   CONV->ReLU       -> CONV->BatchNorm->ReLU (x18)   ->       CONV
                              (64 3x3xc filters)      (64 3x3x64 filters)           (c 3x3x64 filters)
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    num_channels = params.num_channels
    num_filters = 64
    com = comCNN(images, params, num_channels, num_filters)
    rec = recCNN(com, params, num_channels, num_filters)
    

    return com, rec


def model_fn(mode, inputs, params, reuse=False):
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
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        _, logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
