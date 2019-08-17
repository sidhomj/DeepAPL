import tensorflow as tf
import numpy as np

class graph_object(object):
    def __init__(self):
        self.init=0

def Conv_Model(GO,self,kernel_size=(2,2),strides=(2,2),l1_units=12,l2_units=24,l3_units=32):
    # Setup Placeholders
    GO.X = tf.placeholder(tf.float32, [None, self.imgs.shape[1], self.imgs.shape[2], self.imgs.shape[3]], name='Input')
    GO.prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

    conv = tf.layers.conv2d(GO.X, filters=l1_units, kernel_size=kernel_size, strides=strides, padding='valid', activation=tf.nn.relu)
    conv = tf.layers.dropout(conv,GO.prob)
    GO.l1 = conv

    conv = tf.layers.conv2d(conv, filters=l2_units, kernel_size=kernel_size, strides=strides, padding='valid',activation=tf.nn.relu)
    conv = tf.layers.dropout(conv,GO.prob)
    GO.l2 = conv

    kernel_size = (4,4)
    strides = (1,1)
    conv = tf.layers.conv2d(conv, filters=l3_units, kernel_size=kernel_size, strides=strides, padding='valid',activation=tf.nn.relu)
    conv = tf.layers.dropout(conv,GO.prob)
    GO.l3 = conv

def Attention_Layer(GO):
    #method 1
    GO.w = tf.layers.dense(GO.l3,GO.Y.shape[1])
    return tf.reduce_mean(GO.w,[1,2])

    # #method 2
    # GO.w = tf.layers.dense(GO.l3,GO.Y.shape[1],lambda x: isru(x, l=0, h=1, a=0, b=0))
    # w = GO.w[:,:,:,:,tf.newaxis]
    # ft = GO.l3[:,:,:,tf.newaxis,:]
    # return tf.squeeze(tf.layers.dense(tf.reduce_mean(w*ft,[1,2]),1),-1)


def isru(x, l=-1, h=1, a=None, b=None, name='isru', axis=-1,a_bounds=4,b_bounds=4):
    if a is None:
        _a = h - l
    else:
        _a = tf.Variable(name=name + '_a', initial_value=np.ones(np.array([_.value for _ in x.shape])[axis]) + a, trainable=True, dtype=tf.float32)
        _a = 2 ** isru(_a, l=-a_bounds, h=a_bounds)

    if b is None:
        _b = 1
    else:
        _b = tf.Variable(name=name + '_b', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + b, trainable=True, dtype=tf.float32)
        _b = (2 ** isru(_b, l=-b_bounds, h=b_bounds))+1

    return l + (((h - l) / 2) * (1 + (x * ((_a + ((x ** 2) ** _b)) ** -(1 / (2 * _b))))))


