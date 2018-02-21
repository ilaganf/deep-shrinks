import tensorflow as tf

def loss_comm_fn():
	loss = .5 * tf.mean_squared_error(labels=input_images, predictions=RECCNN_output)

def loss_recc_fn():
	loss = .5 * tf.mean_squared_error(labels=CommCNN_outputs - input_images, predictions = RECCNN_output)