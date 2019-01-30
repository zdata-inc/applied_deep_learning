import numpy as np
import tensorflow as tf

def conv_layer(X, in_dims, n_kernels, kernel_size, name, activation=None):
    '''
    Sets up a convolutional layer to be used to build a CNN. 
    
    Parameters:
    X: input data for the layer
    in_dims: number of channels incoming to this layer
    n_kernels: number of convolutional kernels to use in the layer
    kernel_size: the size of each kernel as (kernel_size, kernel_size)
    name: the name of the scope to be used with this layer
    activation: the tensorflow nonlinearity to be used for each neuron
    
    Returns:
    Tensorflow graph description representing the constructed layer
    '''
    with tf.name_scope(name):
        init = tf.truncated_normal([kernel_size, kernel_size, in_dims, n_kernels], stddev=0.1)
        W = tf.Variable(init, name='kernel_weights_{}'.format(name))
        b = tf.Variable(tf.constant(0.1, shape=[n_kernels]))
        conv = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME') + b
        if activation is not None:
            return activation(conv)
        else:
            return conv
        
def max_pool(X, pool_size, name):
    '''
    Sets up a pooling layer to be used to build a CNN. 
    
    Parameters:
    X: input data for the layer
    pool_size: the window size for the maxpool operation in (pool_size, pool_size)
    name: the name of the scope to be used with this layer

    Returns:
    Tensorflow graph description representing the constructed layer
    '''
    with tf.name_scope(name):
        return tf.nn.max_pool(X, ksize=[1, pool_size, pool_size, 1], 
                              strides=[1, pool_size, pool_size, 1], padding='SAME')
        
def dense_layer(X, n_units, name, activation=None):
    '''
    Sets up a hidden layer to be used to build a multilayer perceptron. 
    
    Initializes the weights of the neurons using a normal distribution with
    standard deviation equal to 2 / sqrt(input_dimension + number_neurons)
    
    Parameters:
    X: input data for the layer
    n_units: number of neurons to use in the layer
    name: the name of the scope to be used with this layer
    activation: the tensorflow nonlinearity to be used for each neuron
    
    Returns:
    Tensorflow graph description representing the constructed layer
    '''
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_units)
        init = tf.truncated_normal((n_inputs, n_units), stddev=stddev)
        W = tf.Variable(init, name='hidden_weights_{}'.format(name))
        b = tf.Variable(tf.zeros([n_units]), name='bias_{}'.format(name))
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z