import numpy as np
import tensorflow as tf
from helpers.flatten import Flatten

tf.logging.set_verbosity(tf.logging.INFO)

class Features(object):
    def __init__(self, loss_margin=0.01):
        self.loss_margin = loss_margin

        # num_classes = 5
        input_dim = 64
        channels = 3
        hidden_size = 256
        descriptor_size = 16

        graph = {}

        # Input Layer
        graph['input_layer'] = tf.placeholder(tf.float32, shape=[None, input_dim, input_dim, channels])

        # Batch Size
        graph['batch_size'] = tf.placeholder(tf.int32, shape=[])

        # Convolutional Layer #1
        graph['conv1'] = tf.layers.conv2d(
            inputs=graph['input_layer'],
            filters=16,
            kernel_size=[8, 8],
            activation=tf.nn.relu)

        # Pooling Layer #1
        graph['pool1'] = tf.layers.max_pooling2d(inputs=graph['conv1'], pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        graph['conv2'] = tf.layers.conv2d(
            inputs=graph['pool1'],
            filters=7,
            kernel_size=[5, 5],
            activation=tf.nn.relu)
        graph['pool2'] = tf.layers.max_pooling2d(inputs=graph['conv2'], pool_size=[2, 2], strides=2)

        graph['pool2_flat'] = tf.reshape(graph['pool2'], [-1, 7 * 12 * 12])
        
        graph['fc1'] = tf.layers.dense(inputs=graph['pool2_flat'], units=hidden_size, activation=tf.nn.relu)
        
        graph['fc2'] = tf.layers.dense(inputs=graph['fc1'], units=descriptor_size)

        # graph['anchor_features'] = tf.slice(graph['fc2'], [0 * graph['batch_size'], -1], graph['batch_size'])
        # graph['puller_features'] = tf.slice(graph['fc2'], [1 * graph['batch_size'], -1], graph['batch_size'])
        # graph['pusher_features'] = tf.slice(graph['fc2'], [2 * graph['batch_size'], -1], graph['batch_size'])

        graph['anchor_features'] = graph['fc2'][(0 * graph['batch_size']):(1 * graph['batch_size']), :]
        graph['puller_features'] = graph['fc2'][(1 * graph['batch_size']):(2 * graph['batch_size']), :]
        graph['pusher_features'] = graph['fc2'][(2 * graph['batch_size']):(3 * graph['batch_size']), :]

        graph['diff_pos'] = tf.subtract(graph['anchor_features'], graph['puller_features'])
        graph['diff_neg'] = tf.subtract(graph['anchor_features'], graph['pusher_features'])

        graph['diff_pos'] = tf.multiply(graph['diff_pos'], graph['diff_pos'])
        graph['diff_neg'] = tf.multiply(graph['diff_neg'], graph['diff_neg'])

        graph['diff_pos'] = tf.reduce_sum(graph['diff_pos'], axis=1)
        graph['diff_neg'] = tf.reduce_sum(graph['diff_neg'], axis=1)

        graph['loss_pairs'] = graph['diff_pos']
        graph['loss_triplets_ratio'] = 1 - tf.divide(
            graph['diff_neg'], 
            tf.add(
                self.loss_margin,
                graph['diff_pos']
            )
        )
        
        graph['loss_triplets'] = tf.maximum(
            tf.zeros_like(graph['loss_triplets_ratio']),
            graph['loss_triplets_ratio']
        )

        graph['total_loss'] = graph['loss_triplets'] + graph['loss_pairs']
        graph['loss'] = tf.reduce_mean(graph['total_loss'])

        self.graph = graph

        # Tensorflow Saver
        self.saver = tf.train.Saver()
    
    def prepare_input(self, anchors, pullers, pushers):
        """Prepares input for the graph
        
        Arguments:
            anchors {array} -- a numpy array with all anchors in a batch
            pullers {array} -- a numpy array with all pullers in a batch
            pushers {array} -- a numpy array with all pushers in a batch
        """

        assert all([
            anchors.shape[0] == pullers.shape[0],
            pullers.shape[0] == pushers.shape[0],
        ]), "Anchors, Pullers and Pushers "
        
        N = anchors.shape[0]
        X = np.concatenate((anchors, pullers, pushers), axis=0)

        return X, N

    def evaluate_triplet(self, anchors, pullers, pushers, session=None):
        """Generate the features using the forward pass
        
        Arguments:
            anchors {array} -- [batch_size, width, height, channels]
            pullers {array} -- [batch_size, width, height, channels]
            pushers {array} -- [batch_size, width, height, channels]
        
        Returns:
            tuple -- A tuple of features: (
                [batch_size, number_of_features],
                [batch_size, number_of_features],
                [batch_size, number_of_features]
            )
        """
        
        X, N = self.prepare_input(anchors, pullers, pushers)

        if not session:
            session = tf.Session()

        loss = session.run(self.graph['loss'], feed_dict={
            self.graph['input_layer']: X,
            self.graph['batch_size']: N
        })

        if not session:
            session.close()

        return loss
    
    def optimize(self, session, optimizer, anchors, pullers, pushers):
        """Run a tensorflow optimization step
        
        Arguments:
            session {tf.Session} -- A tensorflow sessions
            optimizer {tf.Optimizer} -- A tensorflow optimizer (initialized w/ learning rate)
            anchors {array} -- a numpy array of anchors
            pullers {array} -- a numpy array of pullers
            pushers {array} -- a numpy array of pushers
        """

        X, N = self.prepare_input(anchors, pullers, pushers)
        return session.run(optimizer, feed_dict={
            self.graph['input_layer']: X,
            self.graph['batch_size']: N
        })