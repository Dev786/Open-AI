import tensorflow as tf
import numpy as np


class DeepModel(object):
    def __init__(self):
        self.optimizer = None
        self.learning_rate = None
        self.decay = None
        # self.epochs = None
        # self.iteration = None
        self.batch_size = None

    def define_neural_net_params(self, learning_rate, optimizer='adam'):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.input_data = tf.

    def define_deep_network(self, hidden_nodes_size=[None, 100, 40, 2]):
        self.hidden_nodes = []
        for i in range(1, len(hidden_nodes_size)):
            self.hidden_nodes.append(
                {
                    "weights": tf.Variable(tf.random.truncated_normal(shape=(hidden_nodes_size[i-1], hidden_nodes_size[i]), dtype=tf.float32, stddev=0.05)),
                    "bias": tf.Variable(tf.zeros(shape=hidden_nodes_size[i]))
                }
            )

    def predict(self, sess,input_data):
        self.predicted_output = None
        for i in range(0, len(self.hidden_nodes)):
            if i is 0:
                self.predicted_output = tf.relu(tf.add(tf.matmul(
                    input_data, self.hidden_nodes[i]['weights']), self.hidden_nodes[i]['bias']))
            else:
                self.predicted_output = tf.nn.relu(tf.add(tf.matmul(
                    self.predicted_output, self.hidden_nodes[i]['weights']), self.hidden_nodes[i]['bias']))

        return sess.run(self.predicted_output,feed_dict={})
    def optimize(self, actual):
        self.logits = self.predicted_output
        loss = tf.reduce_mean(self.logits, actual)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)
